import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm
import logging
import time
from datetime import datetime
import random
import timm

# 设置随机种子以确保结果可重现
def set_random_seed(seed=42):
    """
    设置所有随机种子以确保结果可重现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保CUDA操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置Python哈希种子
    os.environ['PYTHONHASHSEED'] = str(seed)

# DataLoader的worker初始化函数
def worker_init_fn(worker_id):
    """
    确保每个DataLoader worker的随机种子不同但可重现
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# 设置详细日志
def setup_detailed_logging(log_dir=None):
    """
    设置日志系统，确保只创建一个日志文件
    """
    # 如果没有指定日志目录，使用当前脚本目录
    if log_dir is None:
        log_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    
    # 获取或创建logger
    logger = logging.getLogger('train_logger')
    
    # 如果已经配置过，直接返回现有的logger
    if logger.handlers:
        # 从第一个FileHandler中获取文件名
        log_filename = None
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_filename = handler.baseFilename
                break
        return logger, log_filename
    
    logger.setLevel(logging.INFO)
    
    # 创建唯一的日志文件名
    log_filename = os.path.join(log_dir, f'train_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建详细格式
    detailed_formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(detailed_formatter)
    console_handler.setFormatter(detailed_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # 记录日志文件位置
    logger.info(f"日志文件保存位置: {log_filename}")
    
    return logger, log_filename

# 全局logger变量，将在main函数中初始化
logger = None
log_file_path = None

# 配置函数
def get_config():
    """
    返回训练配置参数
    可以在这里修改各种路径和参数
    """
    config = {
        # 路径配置
        'base_dir': r'D:\09_Project\EdgeDeepfakeDetection',
        'log_dir': r'D:\09_Project\EdgeDeepfakeDetection\experiments\baseline\swin_transformer',
        'model_save_dir': r'D:\09_Project\EdgeDeepfakeDetection\models\baseline\swin_transformer\seed42_finetune_93',
        
        # 随机种子配置
        'random_seed': 42,  # 设为None则不固定种子，使用随机结果；设为整数则确保结果可重现
        
        # 训练参数
        'batch_size': 32,  # Swin-T 相对于 MobileNet 更大，需要减小批次大小
        'num_workers': 4,
        'num_epochs': 75,  # 增加到75轮，给模型更多时间优化
        'patience': 12,  # 相应增加patience，平衡过拟合与收敛
        'learning_rate': 1e-4,  # 稳定学习率，避免优化不稳定和损失波动
        'weight_decay': 0.05,  # 强正则化，防止过拟合
        
        # 数据参数
        'add_train_noise': True,  # 训练时添加H.264噪声增强鲁棒性
        'add_test_noise': True,    # 测试时是否测试噪声鲁棒性
    }
    return config

# 自定义数据集
class FFDataset(Dataset):
    def __init__(self, csv_file, faces_dir, split, transform=None, add_noise=False):
        self.faces_dir = faces_dir
        self.transform = transform
        self.add_noise = add_noise  # 控制是否添加噪声
        # self.df = pd.read_csv(csv_file)
        self.df = pd.read_csv(csv_file, dtype={'video_id': str})
        # 建立 video_id 前4位到 (label, split, type) 的映射
        video_info_map = {}
        for _, row in self.df.iterrows():
            vid_prefix = str(row['video_id'])[:4]
            type_id = str(row['video_id'])[0]
            video_type = {
                '0': 'original',
                '1': 'deepfakes',
                '2': 'Face2Face',
                '3': 'FaceShifter',
                '4': 'FaceSwap',
                '5': 'NeuralTexture'
            }[type_id]
            video_info_map[vid_prefix] = (row['label'], row['split'], video_type)
        self.image_paths = []
        self.labels = []
        # 遍历所有类别文件夹
        for video_type in os.listdir(faces_dir):
            type_dir = os.path.join(faces_dir, video_type, 'c23')
            if not os.path.isdir(type_dir):
                continue
            for img_name in os.listdir(type_dir):
                vid_prefix = img_name[:4]
                # 判断是否属于当前 split 且类别匹配
                if vid_prefix in video_info_map:
                    label, split_id, type_name = video_info_map[vid_prefix]
                    if split_id == split and type_name == video_type:
                        self.image_paths.append(os.path.join(type_dir, img_name))
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 只在训练时添加噪声，测试时保持原始图像
        if hasattr(self, 'add_noise') and self.add_noise:
            image = add_h264_noise(image, quality=np.random.randint(30, 50))
        if self.transform:
            image = self.transform(image)
        return image, label

def add_h264_noise(image, quality=30):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    return cv2.imdecode(encimg, cv2.IMREAD_COLOR)

# 数据增强
data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# 训练函数（添加早停、checkpoint、Train AUC监控、渐进式微调）
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, patience=5, save_dir=None):
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info("="*60)
    logger.info("开始训练模型")
    logger.info("="*60)
    
    # 记录模型参数信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型总参数数量: {total_params:,}")
    logger.info(f"可训练参数数量: {trainable_params:,}")
    
    # 记录训练配置
    logger.info(f"训练配置:")
    logger.info(f"  - 总epochs: {num_epochs}")
    logger.info(f"  - 早停patience: {patience}")
    logger.info(f"  - 学习率: {optimizer.param_groups[0]['lr']}")
    logger.info(f"  - 权重衰减: {optimizer.param_groups[0]['weight_decay']}")
    logger.info(f"  - 批次大小: {train_loader.batch_size}")
    logger.info(f"  - 训练样本数: {len(train_loader.dataset)}")
    logger.info(f"  - 验证样本数: {len(val_loader.dataset)}")
    logger.info(f"  - 设备: {next(model.parameters()).device}")
    logger.info(f"  - 模型保存目录: {save_dir}")
    logger.info(f"  - PyTorch随机种子: {torch.initial_seed()}")
    if torch.cuda.is_available():
        logger.info(f"  - CUDA随机种子: {torch.cuda.initial_seed()}")
        logger.info(f"  - CUDNN确定性: {torch.backends.cudnn.deterministic}")
        logger.info(f"  - CUDNN基准测试: {torch.backends.cudnn.benchmark}")
    
    best_auc = 0.0
    no_improve = 0
    training_start_time = time.time()
    
    # 定义模型保存路径
    best_model_path = os.path.join(save_dir, 'swin_tiny_best.pth')
    
    # Swin Transformer 不需要冻结特定层，使用较小的学习率即可
    logger.info("Swin Transformer 使用端到端训练，无需冻结层")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        logger.info(f"当前学习率: {current_lr:.6f}")
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        train_preds, train_labels = [], []
        train_start_time = time.time()
        
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 添加梯度监控
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
            # 收集训练预测结果用于计算Train AUC
            with torch.no_grad():
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                train_preds.extend(probs)
                train_labels.extend(labels.cpu().numpy())
        
        train_time = time.time() - train_start_time
        epoch_loss = running_loss / len(train_loader.dataset)
        train_auc = roc_auc_score(train_labels, train_preds)
        train_acc = accuracy_score(train_labels, (np.array(train_preds) > 0.5).astype(int))
        
        # 验证阶段
        val_start_time = time.time()
        model.eval()
        val_preds, val_labels = [], []
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                # 计算验证损失
                val_loss = criterion(outputs, labels)
                val_running_loss += val_loss.item() * inputs.size(0)
                
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                val_preds.extend(probs)
                val_labels.extend(labels.cpu().numpy())
        
        val_time = time.time() - val_start_time
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_auc = roc_auc_score(val_labels, val_preds)
        val_acc = accuracy_score(val_labels, (np.array(val_preds) > 0.5).astype(int))
        
        epoch_time = time.time() - epoch_start_time
        
        # 详细日志记录
        logger.info(f"Epoch {epoch+1} 结果:")
        logger.info(f"  训练 - Loss: {epoch_loss:.6f}, AUC: {train_auc:.4f}, Acc: {train_acc:.4f}")
        logger.info(f"  验证 - Loss: {val_epoch_loss:.6f}, AUC: {val_auc:.4f}, Acc: {val_acc:.4f}")
        logger.info(f"  时间 - 训练: {train_time:.1f}s, 验证: {val_time:.1f}s, 总计: {epoch_time:.1f}s")
        logger.info(f"  学习率: {current_lr:.6f}")

        print(f"Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Train AUC: {train_auc:.4f}, Train Acc: {train_acc:.4f}")
        print(f"           Val Loss: {val_epoch_loss:.4f}, Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}, Time: {epoch_time:.1f}s")

        # 检查是否为最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  *** 新的最佳验证AUC: {best_auc:.4f} - 模型已保存到 {best_model_path} ***")
            no_improve = 0
        else:
            no_improve += 1
            logger.info(f"  验证AUC未改善，连续{no_improve}轮未提升")
            if no_improve >= patience:
                logger.info(f"达到早停条件 (patience={patience})，在epoch {epoch+1}停止训练")
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Checkpoint 每 5 epoch 保存
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f'swin_tiny_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"  Checkpoint已保存: {checkpoint_path}")

        scheduler.step()
    
    total_training_time = time.time() - training_start_time
    logger.info("="*60)
    logger.info(f"训练完成!")
    logger.info(f"最佳验证AUC: {best_auc:.4f}")
    logger.info(f"总训练时间: {total_training_time/3600:.2f} 小时")
    logger.info(f"最佳模型保存位置: {best_model_path}")
    logger.info("="*60)
    
    return best_auc, best_model_path

if __name__ == '__main__':
    # 加载配置
    config = get_config()
    
    # 设置随机种子以确保结果可重现
    if config['random_seed'] is not None:
        set_random_seed(config['random_seed'])
        print(f"已设置随机种子: {config['random_seed']}")
    
    # 初始化日志到指定目录（确保只设置一次）
    logger, log_file_path = setup_detailed_logging(config['log_dir'])
    
    # 程序开始记录
    program_start_time = time.time()
    logger.info("="*80)
    logger.info("Swin Transformer Tiny Deepfake Detection Training")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    # 定义数据路径
    base_dir = config['base_dir']
    metadata_path = os.path.join(base_dir, 'data', 'datasets', 'metadata', 'video_metadata_c23.csv')
    faces_dir = os.path.join(base_dir, 'data', 'preprocessed', 'faces_224')
    
    logger.info("路径配置:")
    logger.info(f"  - 基础目录: {base_dir}")
    logger.info(f"  - 元数据文件: {metadata_path}")
    logger.info(f"  - 人脸数据目录: {faces_dir}")
    logger.info(f"  - 日志保存目录: {config['log_dir']}")
    logger.info(f"  - 模型保存目录: {config['model_save_dir']}")
    
    logger.info("训练参数配置:")
    logger.info(f"  - 随机种子: {config['random_seed'] if config['random_seed'] is not None else '未设置(随机)'}")
    logger.info(f"  - 批次大小: {config['batch_size']}")
    logger.info(f"  - 工作进程数: {config['num_workers']}")
    logger.info(f"  - 训练轮数: {config['num_epochs']}")
    logger.info(f"  - 早停patience: {config['patience']}")
    logger.info(f"  - 学习率: {config['learning_rate']}")
    logger.info(f"  - 权重衰减: {config['weight_decay']}")
    
    # 检查路径是否存在
    if not os.path.exists(metadata_path):
        logger.error(f"元数据文件不存在: {metadata_path}")
        exit(1)
    if not os.path.exists(faces_dir):
        logger.error(f"人脸数据目录不存在: {faces_dir}")
        exit(1)
    
    logger.info("所有路径检查通过 ✓")
    
    # 数据加载（训练时添加噪声，测试时不添加）
    logger.info("正在加载数据集...")
    
    try:
        train_dataset = FFDataset(metadata_path, faces_dir, split=0, transform=data_transforms['train'], add_noise=False)
        val_dataset = FFDataset(metadata_path, faces_dir, split=1, transform=data_transforms['val'], add_noise=False)
        test_dataset = FFDataset(metadata_path, faces_dir, split=2, transform=data_transforms['val'], add_noise=False)
        
        logger.info("数据集加载成功:")
        logger.info(f"  - 训练集样本数: {len(train_dataset)}")
        logger.info(f"  - 验证集样本数: {len(val_dataset)}")
        logger.info(f"  - 测试集样本数: {len(test_dataset)}")
        
        # 统计标签分布
        train_labels = np.array(train_dataset.labels)
        val_labels = np.array(val_dataset.labels)
        test_labels = np.array(test_dataset.labels)
        
        logger.info("标签分布统计:")
        logger.info(f"  - 训练集 Real: {np.sum(train_labels == 0)}, Fake: {np.sum(train_labels == 1)}")
        logger.info(f"  - 验证集 Real: {np.sum(val_labels == 0)}, Fake: {np.sum(val_labels == 1)}")
        logger.info(f"  - 测试集 Real: {np.sum(test_labels == 0)}, Fake: {np.sum(test_labels == 1)}")
        
    except Exception as e:
        logger.error(f"数据集加载失败: {str(e)}")
        exit(1)
    
    # 创建数据生成器（用于可重现性）
    g = torch.Generator()
    if config['random_seed'] is not None:
        g.manual_seed(config['random_seed'])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'],
        worker_init_fn=worker_init_fn if config['random_seed'] is not None else None,
        generator=g if config['random_seed'] is not None else None
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'],
        worker_init_fn=worker_init_fn if config['random_seed'] is not None else None
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'],
        worker_init_fn=worker_init_fn if config['random_seed'] is not None else None
    )
    
    logger.info(f"数据加载器配置: batch_size={config['batch_size']}, num_workers=4")

    # 数据检查：验证数据加载是否正确
    logger.info("正在验证数据加载...")
    sample_batch = next(iter(train_loader))
    sample_inputs, sample_labels = sample_batch
    logger.info(f"样本批次形状: inputs={sample_inputs.shape}, labels={sample_labels.shape}")
    logger.info(f"输入数据统计: mean={sample_inputs.mean():.4f}, std={sample_inputs.std():.4f}")
    logger.info(f"输入数据范围: min={sample_inputs.min():.4f}, max={sample_inputs.max():.4f}")
    logger.info(f"标签分布: {torch.bincount(sample_labels)}")
    
    # 检查数据是否合理
    if sample_inputs.min() < -5 or sample_inputs.max() > 5:
        logger.warning("⚠️ 输入数据范围异常，可能需要检查数据预处理")
    if len(torch.unique(sample_labels)) < 2:
        logger.warning("⚠️ 单个批次中只有一种标签，可能影响训练")
    
    logger.info("✓ 数据验证完成")

    # 加载 Swin Transformer Tiny 模型
    logger.info("正在初始化Swin Transformer Tiny模型...")
    try:
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=2)
        
        # 在分类头添加 Dropout (0.1) - 降低dropout以增强拟合能力
        original_head = model.head
        model.head = nn.Sequential(
            nn.Dropout(0.1),
            original_head
        )
        
        logger.info("已加载预训练的Swin Transformer Tiny模型")
        logger.info("模型结构: swin_tiny_patch4_window7_224 + Dropout(0.1)")
        
        # 验证预训练权重是否正确加载
        logger.info("验证预训练权重加载状态...")
        # 检查第一层的权重统计
        first_layer = None
        for name, param in model.named_parameters():
            if 'patch_embed.proj.weight' in name:
                first_layer = param
                break
        
        if first_layer is not None:
            logger.info(f"第一层权重统计: mean={first_layer.mean().item():.6f}, std={first_layer.std().item():.6f}")
            if abs(first_layer.mean().item()) < 1e-6 and abs(first_layer.std().item() - 0) < 1e-6:
                logger.warning("警告：权重可能未正确初始化！")
            else:
                logger.info("✓ 权重已正确初始化")
        
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        logger.info("尝试安装timm包...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'timm'])
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=2)
        
        # 在分类头添加 Dropout (0.1) - 降低dropout以增强拟合能力
        original_head = model.head
        model.head = nn.Sequential(
            nn.Dropout(0.1),
            original_head
        )
        
        logger.info("已成功加载Swin Transformer Tiny模型 + Dropout(0.1)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA设备信息: {torch.cuda.get_device_name()}")
        logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    model = model.to(device)

    # 损失和优化器（AdamW 对 Transformer 更友好）
    criterion = nn.CrossEntropyLoss()  # 移除标签平滑以追求更低Loss
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑已注释，追求更好拟合
    # 使用warmup scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    # 使用warmup + cosine annealing scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    
    # warmup scheduler - 缩短至2 epochs，更快进入主要学习阶段
    warmup_epochs = 2
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=config['num_epochs'], eta_min=1e-5)  # 提高最小学习率至1e-5
    scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], milestones=[warmup_epochs])
    
    logger.info("优化器配置:")
    logger.info(f"  - 优化器: AdamW")
    logger.info(f"  - 初始学习率: {config['learning_rate']}")
    logger.info(f"  - 最小学习率: 1e-5 (提高以保持后期优化能力)")
    logger.info(f"  - 权重衰减: {config['weight_decay']}")
    logger.info(f"  - 调度器: Warmup({warmup_epochs}) + CosineAnnealingLR(T_max={config['num_epochs']})")
    logger.info(f"  - 损失函数: CrossEntropyLoss (移除label_smoothing追求更低Loss)")
    logger.info(f"  - 分类头Dropout: 0.1 (降低以增强拟合能力)")

    # 训练
    best_auc, best_model_path = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                                          num_epochs=config['num_epochs'], patience=config['patience'], 
                                          save_dir=config['model_save_dir'])
    print(f"Best Validation AUC: {best_auc:.4f}")
    logger.info(f"训练结束 - 最佳验证AUC: {best_auc:.4f}")

    # 测试（纯净性能，无噪声）
    logger.info("\n" + "="*60)
    logger.info("开始测试阶段 - 纯净性能（无H.264噪声）")
    logger.info("="*60)
    
    print("\n=== 测试纯净性能（无H.264噪声） ===")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    test_start_time = time.time()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, desc="测试进度")):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            test_preds.extend(probs)
            test_labels.extend(labels.cpu().numpy())
    
    test_time = time.time() - test_start_time
    test_auc = roc_auc_score(test_labels, test_preds)
    test_acc = accuracy_score(test_labels, (np.array(test_preds) > 0.5).astype(int))
    
    logger.info(f"纯净测试结果:")
    logger.info(f"  - 测试AUC: {test_auc:.4f}")
    logger.info(f"  - 测试准确率: {test_acc:.4f}")
    logger.info(f"  - 测试时间: {test_time:.2f}秒")
    logger.info(f"  - 测试样本数: {len(test_labels)}")
    
    print(f"Test AUC (Clean): {test_auc:.4f}, Test Acc (Clean): {test_acc:.4f}")
    
    # 测试有噪声性能（可选）
    logger.info("\n" + "="*60)
    logger.info("开始测试阶段 - 噪声鲁棒性（含H.264噪声）")
    logger.info("="*60)
    
    print("\n=== 测试噪声鲁棒性（含H.264噪声） ===")
    test_dataset_noisy = FFDataset(metadata_path, faces_dir, split=2, transform=data_transforms['val'], add_noise=True)
    test_loader_noisy = DataLoader(
        test_dataset_noisy, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'],
        worker_init_fn=worker_init_fn if config['random_seed'] is not None else None
    )
    
    test_noisy_start_time = time.time()
    test_preds_noisy, test_labels_noisy = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader_noisy, desc="噪声测试进度"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            test_preds_noisy.extend(probs)
            test_labels_noisy.extend(labels.cpu().numpy())
    
    test_noisy_time = time.time() - test_noisy_start_time
    test_auc_noisy = roc_auc_score(test_labels_noisy, test_preds_noisy)
    test_acc_noisy = accuracy_score(test_labels_noisy, (np.array(test_preds_noisy) > 0.5).astype(int))
    
    logger.info(f"噪声测试结果:")
    logger.info(f"  - 测试AUC (噪声): {test_auc_noisy:.4f}")
    logger.info(f"  - 测试准确率 (噪声): {test_acc_noisy:.4f}")
    logger.info(f"  - 测试时间: {test_noisy_time:.2f}秒")
    logger.info(f"  - 鲁棒性评估 (AUC下降): {test_auc - test_auc_noisy:.4f}")
    
    print(f"Test AUC (Noisy): {test_auc_noisy:.4f}, Test Acc (Noisy): {test_acc_noisy:.4f}")
    
    # 程序结束统计
    total_program_time = time.time() - program_start_time
    logger.info("\n" + "="*80)
    logger.info("程序执行完成")
    logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"总执行时间: {total_program_time/3600:.2f} 小时")
    logger.info("\n最终结果总结:")
    logger.info(f"  - 最佳验证AUC: {best_auc:.4f}")
    logger.info(f"  - 纯净测试AUC: {test_auc:.4f}")
    logger.info(f"  - 噪声测试AUC: {test_auc_noisy:.4f}")
    logger.info(f"  - 噪声鲁棒性: {((test_auc_noisy/test_auc)*100):.1f}%")
    logger.info("="*80)