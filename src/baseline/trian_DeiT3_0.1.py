import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, roc_curve
import pandas as pd
import numpy as np
import cv2
import os
import sys
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
    """DeiT3 Base (deit3_base_patch16_224, ~89M) 在 4090 (24GB) + CUDA12.1 微调深度伪造检测推荐配置"""
    config = {
        # 基础路径（按需修改）。保留 Linux/Ubuntu 训练路径；在 Windows 上测试可自行替换。
        'base_dir': r'/root/EdgeDeepfakeDetection',
        'log_dir': r'/root/EdgeDeepfakeDetection/experiments/baseline/deit3_base',
        'model_save_dir': r'/root/EdgeDeepfakeDetection/models/baseline/deit3_base/seed42',

        # 随机种子
        'random_seed': 42,

        # 4090 显存允许更大 batch；使用 AMP 后 128 基本稳定，可再尝试 160/192
        'batch_size': 128,
        'num_workers': 8,
    # DataLoader 性能参数（之前引用但未在配置中定义，补充）
    'pin_memory': True,
    'prefetch_factor': 4,          # CPU 合理负载；如磁盘/CPU 充足可调 6
    'persistent_workers': True,
        'num_epochs': 40,
        'patience': 8,
        'learning_rate': 2e-4,        # 对应有效 batch 128 的微调 lr（线性放缩）
        'weight_decay': 0.05,

        # 训练策略
        'gradient_accumulation_steps': 1,  # 若需要有效 batch 192，可设 batch=96 & steps=2
        'mixed_precision': True,           # AMP fp16；如想用 bf16 可改 amp_dtype 并关闭 scaler 分支
        'amp_dtype': 'fp16',
        'gradient_clip_norm': 1.0,
        'compile_model': True,             # torch>=2.0 可加速
        'channels_last': False,            # ViT 不必使用 channels_last

        # 数据噪声评估策略
        'add_train_noise': False,          # 训练不加编码噪声，避免破坏伪造纹理
        'add_test_noise': True,            # 测试阶段增加一组噪声鲁棒性评估

        # 调度器（warmup + cosine）
        'warmup_epochs': 5,
        'cosine_eta_min': 1e-6,
        'cosine_restart': False,

        # 正则化
        'dropout_rate': 0.0,
        'stochastic_depth_rate': 0.1,      # drop path；过拟合可升 0.15~0.2
        'label_smoothing': 0.1,

        # 可选：层级衰减（ViT 可启用，但微调二分类通常关闭）
        'use_layer_decay': False,
        'layer_decay': 0.75,

        # EMA（初次可关闭）
        'use_ema': False,
        'ema_decay': 0.9998,

        # Mixup / CutMix（微调默认关闭；需要时开启）
        'use_mixup': False,
        'mixup_alpha': 0.2,
        'cutmix_alpha': 0.0,

        # 保存 & 日志
        'save_freq': 5,
        'eval_freq': 1,
        'log_freq': 100,
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
        
        # 确保标签是整数类型，用于CrossEntropyLoss
        label = int(label)
        return image, label

def add_h264_noise(image, quality=30):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    return cv2.imdecode(encimg, cv2.IMREAD_COLOR)

def ensure_2d_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    将模型输出规范为 (N, C) 形状：
    - 对于 4D/N>2 的输出，先在空间维度上做平均池化，再展平成 (N, C)
    - 对于已经是 2D 的输出，直接返回
    这样可确保与 CrossEntropyLoss 的 (N, C) x (N,) 目标匹配。
    """
    if logits.dim() == 2:
        return logits
    if logits.dim() > 2:
        # 折叠除 (N, C) 外的其余维度，并在这些维度上取平均
        # 例如 (N, C, H, W) -> (N, C)
        return torch.flatten(logits, start_dim=2).mean(dim=2)
    return logits

def calculate_eer(y_true, y_scores):
    """
    计算EER (Equal Error Rate)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    
    # 找到FPR和FNR最接近的点
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    
    return eer, eer_threshold

# ConvNeXt V2 优化的数据增强
def get_data_transforms(target_size=224):
    """DeiT3 Base 推荐的数据增强（保持伪造细节，降低过激形变）"""
    return {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(target_size, scale=(0.9, 1.0), ratio=(0.98, 1.02)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.5, 2.0))
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(int(target_size * 1.14)),  # 256 -> 224 center crop 近似
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

# 训练函数（V100 32GB + ConvNeXt V2 深度优化版本）
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler=None, 
                num_epochs=50, patience=5, save_dir=None, gradient_accumulation_steps=1, 
                gradient_clip_norm=0, channels_last=False):
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
    best_model_path = os.path.join(save_dir, 'deit3_base_best.pth')
    
    # DeiT3 微调
    logger.info("DeiT3 Base 使用端到端微调（不冻结层）")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        logger.info(f"当前学习率: {current_lr:.6f}")
        
        # 训练阶段 - V100优化版本
        model.train()
        running_loss = 0.0
        train_preds, train_labels = [], []
        train_start_time = time.time()
        
        # 梯度累积优化
        optimizer.zero_grad()
        
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 混合精度训练
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    # 支持channels_last优化
                    if channels_last:
                        inputs = inputs.to(memory_format=torch.channels_last)
                    outputs = model(inputs)
                    outputs = ensure_2d_logits(outputs)
                    loss = criterion(outputs, labels)
                    loss = loss / gradient_accumulation_steps  # 标准化梯度累积的损失
                
                scaler.scale(loss).backward()
                
                # 梯度累积
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # 梯度裁剪（使用配置参数）
                    scaler.unscale_(optimizer)
                    grad_norm = 0
                    if gradient_clip_norm > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # 标准训练
                if channels_last:
                    inputs = inputs.to(memory_format=torch.channels_last)
                outputs = model(inputs)
                outputs = ensure_2d_logits(outputs)
                loss = criterion(outputs, labels)
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                # 梯度累积
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    grad_norm = 0
                    if gradient_clip_norm > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
                    optimizer.step()
                    optimizer.zero_grad()
            
            running_loss += loss.item() * inputs.size(0) * gradient_accumulation_steps  # 恢复真实损失
            
            # 收集训练预测结果用于计算Train AUC
            with torch.no_grad():
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                else:
                    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                train_preds.extend(probs)
                train_labels.extend(labels.cpu().numpy())
        
        train_time = time.time() - train_start_time
        epoch_loss = running_loss / len(train_loader.dataset)
        train_auc = roc_auc_score(train_labels, train_preds)
        train_acc = accuracy_score(train_labels, (np.array(train_preds) > 0.5).astype(int))
        
        # 验证阶段 - V100优化版本
        val_start_time = time.time()
        model.eval()
        val_preds, val_labels = [], []
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 混合精度推理
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        outputs = ensure_2d_logits(outputs)
                        val_loss = criterion(outputs, labels)
                        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                else:
                    outputs = model(inputs)
                    outputs = ensure_2d_logits(outputs)
                    val_loss = criterion(outputs, labels)
                    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                
                val_running_loss += val_loss.item() * inputs.size(0)
                val_preds.extend(probs)
                val_labels.extend(labels.cpu().numpy())
        
        val_time = time.time() - val_start_time
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_auc = roc_auc_score(val_labels, val_preds)
        val_acc = accuracy_score(val_labels, (np.array(val_preds) > 0.5).astype(int))
        val_ap = average_precision_score(val_labels, val_preds)
        val_eer, val_eer_threshold = calculate_eer(val_labels, val_preds)
        
        epoch_time = time.time() - epoch_start_time
        
        # 详细日志记录
        logger.info(f"Epoch {epoch+1} 结果:")
        train_ap = average_precision_score(train_labels, train_preds)
        train_eer, train_eer_threshold = calculate_eer(train_labels, train_preds)
        logger.info(f"  训练 - Loss: {epoch_loss:.6f}, AUC: {train_auc:.4f}, AP: {train_ap:.4f}, EER: {train_eer:.4f}, Acc: {train_acc:.4f}")
        logger.info(f"  验证 - Loss: {val_epoch_loss:.6f}, AUC: {val_auc:.4f}, AP: {val_ap:.4f}, EER: {val_eer:.4f}, Acc: {val_acc:.4f}")
        logger.info(f"  时间 - 训练: {train_time:.1f}s, 验证: {val_time:.1f}s, 总计: {epoch_time:.1f}s")
        logger.info(f"  学习率: {current_lr:.6f}")

        print(f"Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Train AUC: {train_auc:.4f}, Train AP: {train_ap:.4f}, Train EER: {train_eer:.4f}, Train Acc: {train_acc:.4f}")
        print(f"           Val Loss: {val_epoch_loss:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}, Val EER: {val_eer:.4f}, Val Acc: {val_acc:.4f}, Time: {epoch_time:.1f}s")

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
            checkpoint_path = os.path.join(save_dir, f'convnextv2_base_epoch_{epoch+1}.pth')
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
    logger.info("DeiT3 Base Deepfake Detection Training")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    # 记录完整的配置字典到日志文件
    logger.info("完整配置参数字典:")
    logger.info("-" * 50)
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("-" * 50)
    
    # 记录系统环境信息
    logger.info("系统环境信息:")
    logger.info(f"  - Python版本: {sys.version}")
    logger.info(f"  - PyTorch版本: {torch.__version__}")
    logger.info(f"  - CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  - CUDA版本: {torch.version.cuda}")
        logger.info(f"  - GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    logger.info(f"  - 工作目录: {os.getcwd()}")
    logger.info(f"  - 脚本路径: {os.path.abspath(__file__)}")
    logger.info(f"  - 日志文件: {log_file_path}")
    logger.info("="*80)
    
    # 定义数据路径
    base_dir = config['base_dir']
    metadata_path = os.path.join(base_dir, 'data', 'dataset', 'FFpp', 'video_metadata_c23.csv')
    faces_dir = os.path.join(base_dir, 'data', 'preprocessed', 'FFpp', 'faces_256')
    
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
    
    # 记录完整的深度优化配置参数
    logger.info("4090 + DeiT3 优化配置:")
    logger.info(f"  - 梯度累积步数: {config['gradient_accumulation_steps']}")
    logger.info(f"  - 混合精度训练: {config['mixed_precision']}")
    logger.info(f"  - Pin内存: {config['pin_memory']}")
    logger.info(f"  - 预取因子: {config['prefetch_factor']}")
    logger.info(f"  - 持久化Workers: {config['persistent_workers']}")
    logger.info(f"  - 模型编译优化: {config.get('compile_model', False)}")
    logger.info(f"  - Channels Last格式: {config.get('channels_last', False)}")
    
    logger.info("学习率调度优化:")
    logger.info(f"  - Warmup轮数: {config['warmup_epochs']}")
    logger.info(f"  - 余弦最小学习率: {config['cosine_eta_min']}")
    logger.info(f"  - 余弦重启: {config.get('cosine_restart', False)}")
    
    logger.info("正则化配置:")
    logger.info(f"  - Dropout率: {config['dropout_rate']}")
    logger.info(f"  - 随机深度率: {config.get('stochastic_depth_rate', 0)}")
    logger.info(f"  - 标签平滑: {config['label_smoothing']}")
    logger.info(f"  - 梯度裁剪范数: {config.get('gradient_clip_norm', 0)}")
    
    logger.info("高级优化配置:")
    logger.info(f"  - 混合精度数据类型: {config.get('amp_dtype', 'float16')}")
    logger.info(f"  - 层级学习率衰减: {config.get('use_layer_decay', False)}")
    logger.info(f"  - 层级衰减率: {config.get('layer_decay', 0.65)}")
    logger.info(f"  - EMA: {config.get('use_ema', False)}")
    logger.info(f"  - EMA衰减率: {config.get('ema_decay', 0.9999)}")
    
    logger.info("数据配置:")
    logger.info(f"  - 训练时添加噪声: {config['add_train_noise']}")
    logger.info(f"  - 测试时添加噪声: {config['add_test_noise']}")
    
    logger.info("保存和记录配置:")
    logger.info(f"  - 保存频率: {config.get('save_freq', 5)} epochs")
    logger.info(f"  - 验证频率: {config.get('eval_freq', 1)} epochs")
    logger.info(f"  - 日志记录频率: {config.get('log_freq', 100)} steps")
    
    # 计算和记录有效配置
    effective_batch_size = config['batch_size'] * config['gradient_accumulation_steps']
    logger.info("有效训练配置:")
    logger.info(f"  - 有效批次大小: {effective_batch_size}")
    logger.info(f"  - 预估单步GPU内存: {config['batch_size'] * 3 * 224 * 224 * 4 / (1024**3):.2f} GB")
    logger.info(f"  - 预估总GPU内存需求: ~{config['batch_size'] * 3 * 224 * 224 * 4 * 3 / (1024**3):.1f} GB")
    
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
    
    # 先创建一个临时的数据变换来确定输入尺寸
    temp_transforms = get_data_transforms(224)  # ConvNeXt V2 默认使用 224
    
    try:
        train_dataset = FFDataset(metadata_path, faces_dir, split=0, transform=temp_transforms['train'], add_noise=False)
        val_dataset = FFDataset(metadata_path, faces_dir, split=1, transform=temp_transforms['val'], add_noise=False)
        test_dataset = FFDataset(metadata_path, faces_dir, split=2, transform=temp_transforms['val'], add_noise=False)
        
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
    
    # 自定义collate函数，确保标签是正确的类型
    def custom_collate_fn(batch):
        """
        自定义collate函数，确保标签是长整型，适用于CrossEntropyLoss
        """
        images, labels = zip(*batch)
        images = torch.stack(images, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)  # 明确转换为长整型
        return images, labels
    
    # V100 32GB + ConvNeXt V2 Base 优化的数据加载器配置
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'],
        worker_init_fn=worker_init_fn if config['random_seed'] is not None else None,
        generator=g if config['random_seed'] is not None else None,
        pin_memory=config['pin_memory'],           # V100优化：加速GPU传输
        prefetch_factor=config['prefetch_factor'], # V100优化：预取数据
        persistent_workers=config['persistent_workers'], # V100优化：保持worker进程
        drop_last=True,                           # 确保batch size一致，优化训练稳定性
        collate_fn=custom_collate_fn              # 使用自定义collate函数
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'],
        worker_init_fn=worker_init_fn if config['random_seed'] is not None else None,
        pin_memory=config['pin_memory'],
        prefetch_factor=config['prefetch_factor'],
        persistent_workers=config['persistent_workers'],
        drop_last=False,  # 验证时不丢弃，确保所有数据都被评估
        collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'],
        worker_init_fn=worker_init_fn if config['random_seed'] is not None else None,
        pin_memory=config['pin_memory'],
        prefetch_factor=config['prefetch_factor'],
        persistent_workers=config['persistent_workers'],
        drop_last=False,  # 测试时不丢弃，确保所有数据都被测试
        collate_fn=custom_collate_fn
    )
    
    logger.info(f"4090 + DeiT3 数据加载器配置:")
    logger.info(f"  - batch_size={config['batch_size']} (AMP 下 24GB 可承载的推荐值)")
    logger.info(f"  - num_workers={config['num_workers']} (充分利用CPU多核资源)")
    logger.info(f"  - pin_memory={config['pin_memory']} (加速GPU传输)")
    logger.info(f"  - prefetch_factor={config['prefetch_factor']} (预取优化)")
    logger.info(f"  - persistent_workers={config['persistent_workers']} (减少worker重启开销)")
    logger.info(f"  - drop_last=True (训练时保证batch一致性)")
    
    # 计算预期内存使用
    estimated_memory_per_batch = config['batch_size'] * 3 * 224 * 224 * 4 / (1024**3)  # FP32（224输入）
    estimated_model_memory = 89_000_000 * 4 / (1024**3)  # 89M参数的FP32内存
    logger.info(f"内存使用估计:")
    logger.info(f"  - 单批次数据内存: {estimated_memory_per_batch:.2f} GB")
    logger.info(f"  - 模型参数内存: {estimated_model_memory:.2f} GB")
    logger.info(f"  - 总预估内存: ~{estimated_memory_per_batch * 2 + estimated_model_memory * 3:.1f} GB (包含梯度和优化器)")
    logger.info(f"  - 4090 24GB 预计余量可尝试更大 batch 或开启 EMA")

    # 数据检查：验证数据加载是否正确
    logger.info("正在验证数据加载...")
    sample_batch = next(iter(train_loader))
    sample_inputs, sample_labels = sample_batch
    logger.info(f"样本批次形状: inputs={sample_inputs.shape}, labels={sample_labels.shape}")
    logger.info(f"输入数据统计: mean={sample_inputs.mean():.4f}, std={sample_inputs.std():.4f}")
    logger.info(f"输入数据范围: min={sample_inputs.min():.4f}, max={sample_inputs.max():.4f}")
    
    # 详细检查标签
    logger.info(f"标签统计:")
    logger.info(f"  - 标签数据类型: {sample_labels.dtype}")
    logger.info(f"  - 标签形状: {sample_labels.shape}")
    logger.info(f"  - 标签范围: min={sample_labels.min()}, max={sample_labels.max()}")
    logger.info(f"  - 标签分布: {torch.bincount(sample_labels)}")
    logger.info(f"  - 标签唯一值: {torch.unique(sample_labels)}")
    
    # 验证标签是否适用于CrossEntropyLoss
    if len(sample_labels.shape) != 1:
        logger.error(f"错误：标签维度应该是1维，但得到了 {len(sample_labels.shape)} 维")
    if sample_labels.dtype not in [torch.int64, torch.long]:
        logger.warning(f"警告：标签数据类型是 {sample_labels.dtype}，CrossEntropyLoss期望 int64/long类型")
    
    # 检查数据是否合理
    if sample_inputs.min() < -5 or sample_inputs.max() > 5:
        logger.warning("⚠️ 输入数据范围异常，可能需要检查数据预处理")
    if len(torch.unique(sample_labels)) < 2:
        logger.warning("⚠️ 单个批次中只有一种标签，可能影响训练")
    
    logger.info("✓ 数据验证完成")

    # 加载 ConvNeXt V2 Base 模型
    logger.info("正在初始化 DeiT3 Base 模型...")

    model = None

    # 优先加载本地下载的预训练权重 (quick_download_models.py 保存格式)
    local_weight_path = os.path.join(
        config['base_dir'], 'models', 'pretrained_models', 'deit3_base_patch16_224', 'deit3_base_patch16_224.pth'
    )
    if not os.path.exists(local_weight_path):
        logger.error(f"本地预训练权重不存在: {local_weight_path}")
        logger.error("当前配置指定仅使用本地权重，终止运行。")
        raise FileNotFoundError(local_weight_path)

    logger.info(f"使用方案1严格加载本地 DeiT3 预训练权重: {local_weight_path}")
    # 1) 先按 1000 类构建，与 checkpoint 对齐
    model = timm.create_model(
        'deit3_base_patch16_224',
        pretrained=False,
        num_classes=1000,
        drop_path_rate=config.get('stochastic_depth_rate', 0.1),
        drop_rate=config['dropout_rate']
    )
    # 2) 严格加载
    state_dict = torch.load(local_weight_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    logger.info("✓ 严格加载完成 (1000 类权重匹配)")
    # 3) 重置分类头到二分类
    if hasattr(model, 'reset_classifier'):
        model.reset_classifier(2)
        logger.info("✓ 调整分类头为 2 类 (reset_classifier)")
    else:
        in_features = model.head.in_features if hasattr(model, 'head') else model.get_classifier().in_features
        if hasattr(model, 'head'):
            model.head = nn.Linear(in_features, 2)
        else:
            model.classifier = nn.Linear(in_features, 2)
        logger.info("✓ 手动替换分类头为 2 类")

    logger.info("模型详细信息:")
    logger.info(f"  - 架构: DeiT3 Base")
    logger.info(f"  - 输入分辨率: 224x224")
    logger.info(f"  - 参数量: ~89M")

    # 打印第一层卷积权重统计
    logger.info("验证模型权重初始化状态...")
    first_weight = None
    for name, param in model.named_parameters():
        if name.endswith('weight') and param.dim() == 4:
            first_weight = param
            break
    if first_weight is not None:
        logger.info(f"第一层卷积权重统计: mean={first_weight.mean().item():.6f}, std={first_weight.std().item():.6f}")

    # 记录详细的模型架构信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型架构详情:")
    logger.info(f"  - 总参数数量: {total_params:,}")
    logger.info(f"  - 可训练参数数量: {trainable_params:,}")
    logger.info(f"  - 模型大小估计: {total_params * 4 / 1024 / 1024:.1f} MB (FP32)")

    # DeiT3 使用 224x224 输入
    target_size = 224
    logger.info(f"使用 DeiT3 默认输入尺寸: {target_size}x{target_size}")

    # 使用现有的数据变换
    data_transforms = temp_transforms
    logger.info("使用默认输入尺寸224x224")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA设备信息: {torch.cuda.get_device_name()}")
        logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"cuDNN版本: {torch.backends.cudnn.version()}")
        logger.info(f"Tensor Core支持: {'是' if torch.cuda.get_device_capability()[0] >= 7 else '否'}")
    
    # 4090 + DeiT3 训练优化
    model = model.to(device)
    
    # ViT 一般无需 channels_last（默认关闭，保留选项）
    if config.get('channels_last', False):
        model = model.to(memory_format=torch.channels_last)
        logger.info("✓ 启用channels_last内存格式优化")
    
    # 尝试启用PyTorch编译优化（如果PyTorch版本支持）
    if config.get('compile_model', False) and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            logger.info("✓ 启用PyTorch 2.0编译优化")
        except Exception as e:
            logger.warning(f"PyTorch编译优化失败: {e}")

    # 损失与优化器
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config['label_smoothing']
    )
    logger.info(f"损失函数: CrossEntropyLoss (标签平滑={config['label_smoothing']})")
    
    # 参数分组（不对 LN / bias 施加 weight decay）
    def build_vit_param_groups(m, base_lr, wd):
        decay, no_decay = [], []
        for n, p in m.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() == 1 or n.endswith('.bias') or 'norm' in n.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        groups = [
            {'params': decay, 'weight_decay': wd, 'lr': base_lr},
            {'params': no_decay, 'weight_decay': 0.0, 'lr': base_lr}
        ]
        return groups

    if config.get('use_layer_decay', False):
        # 简化实现：为每层 blocks[i] 设置 layer decay（深度 12）
        layer_decay = config.get('layer_decay', 0.75)
        param_groups = []
        # 收集 block 层次
        depth = 0
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # 估算所在层深度：blocks.X
            lr_scale = 1.0
            if 'blocks.' in n:
                try:
                    idx = int(n.split('blocks.')[1].split('.')[0])
                    depth = idx
                except Exception:
                    depth = 0
                lr_scale = layer_decay ** (model.blocks.__len__() - 1 - depth)
            elif 'head' in n:
                lr_scale = 1.0
            if p.dim() == 1 or n.endswith('.bias') or 'norm' in n.lower():
                wd = 0.0
            else:
                wd = config['weight_decay']
            param_groups.append({'params': [p], 'weight_decay': wd, 'lr': config['learning_rate'] * lr_scale})
        optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
        logger.info(f"✓ 使用层级学习率衰减 layer_decay={layer_decay}")
    else:
        optimizer = optim.AdamW(
            build_vit_param_groups(model, config['learning_rate'], config['weight_decay']),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    # 添加梯度裁剪（防止大模型梯度爆炸）
    if config.get('gradient_clip_norm', 0) > 0:
        logger.info(f"✓ 启用梯度裁剪 (max_norm={config['gradient_clip_norm']})")
    
    # 学习率调度器（warmup + cosine）
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    
    warmup_epochs = config['warmup_epochs']
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    
    if config.get('cosine_restart', False):
        # 使用带重启的余弦退火
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        main_scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=config['num_epochs'] - warmup_epochs,
            eta_min=config['cosine_eta_min']
        )
    else:
        # 使用标准余弦退火
        main_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=config['num_epochs'] - warmup_epochs, 
            eta_min=config['cosine_eta_min']
        )
    
    scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], milestones=[warmup_epochs])
    
    # 混合精度训练配置（V100 Tensor Core优化）
    scaler = None
    if config['mixed_precision']:
        scaler = torch.cuda.amp.GradScaler()
        logger.info(f"✓ 混合精度训练已启用 (dtype={config.get('amp_dtype', 'float16')})")
    
    logger.info("DeiT3 Base 训练配置:")
    logger.info(f"  - 优化器: AdamW {'+ LayerDecay' if config.get('use_layer_decay', False) else ''}")
    logger.info(f"  - 基础学习率: {config['learning_rate']}")
    logger.info(f"  - 最小学习率: {config['cosine_eta_min']}")
    logger.info(f"  - 权重衰减: {config['weight_decay']}")
    logger.info(f"  - Warmup轮数: {warmup_epochs}")
    logger.info(f"  - 调度器: {'CosineRestart' if config.get('cosine_restart', False) else 'Cosine'}")
    logger.info(f"  - 标签平滑: {config['label_smoothing']}")
    logger.info(f"  - Dropout率: {config['dropout_rate']}")
    logger.info(f"  - DropPath率: {config.get('stochastic_depth_rate', 0)}")
    logger.info(f"  - 梯度裁剪: {config.get('gradient_clip_norm', '关闭')}")
    logger.info(f"  - 混合精度: {'开启' if config['mixed_precision'] else '关闭'}")
    logger.info(f"  - 梯度累积: {config['gradient_accumulation_steps']}")
    logger.info(f"  - 有效batch size: {config['batch_size'] * config['gradient_accumulation_steps']}")
    
    # 内存和性能监控
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 清理GPU缓存
        current_memory = torch.cuda.memory_allocated() / 1024**3
        max_memory = torch.cuda.max_memory_allocated() / 1024**3
        logger.info(f"GPU内存状态: 当前 {current_memory:.2f}GB, 峰值 {max_memory:.2f}GB")

    # 开始训练 DeiT3 Base
    best_auc, best_model_path = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, 
        scaler=scaler,
        num_epochs=config['num_epochs'], 
        patience=config['patience'], 
        save_dir=config['model_save_dir'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        gradient_clip_norm=config.get('gradient_clip_norm', 0),
        channels_last=config.get('channels_last', False)
    )
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
            
            # 混合精度推理 + channels_last优化
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    if config.get('channels_last', False):
                        inputs = inputs.to(memory_format=torch.channels_last)
                    outputs = model(inputs)
                    outputs = ensure_2d_logits(outputs)
                    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            else:
                if config.get('channels_last', False):
                    inputs = inputs.to(memory_format=torch.channels_last)
                outputs = model(inputs)
                outputs = ensure_2d_logits(outputs)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            
            test_preds.extend(probs)
            test_labels.extend(labels.cpu().numpy())
    
    test_time = time.time() - test_start_time
    test_auc = roc_auc_score(test_labels, test_preds)
    test_acc = accuracy_score(test_labels, (np.array(test_preds) > 0.5).astype(int))
    test_ap = average_precision_score(test_labels, test_preds)
    test_eer, test_eer_threshold = calculate_eer(test_labels, test_preds)
    
    logger.info(f"纯净测试结果:")
    logger.info(f"  - 测试AUC: {test_auc:.4f}")
    logger.info(f"  - 测试AP: {test_ap:.4f}")
    logger.info(f"  - 测试EER: {test_eer:.4f} (阈值: {test_eer_threshold:.4f})")
    logger.info(f"  - 测试准确率: {test_acc:.4f}")
    logger.info(f"  - 测试时间: {test_time:.2f}秒")
    logger.info(f"  - 测试样本数: {len(test_labels)}")
    
    print(f"Test AUC (Clean): {test_auc:.4f}, Test AP (Clean): {test_ap:.4f}, Test EER (Clean): {test_eer:.4f}, Test Acc (Clean): {test_acc:.4f}")
    
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
        worker_init_fn=worker_init_fn if config['random_seed'] is not None else None,
        collate_fn=custom_collate_fn  # 使用同样的自定义collate函数
    )
    
    test_noisy_start_time = time.time()
    test_preds_noisy, test_labels_noisy = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader_noisy, desc="噪声测试进度"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 混合精度推理 + channels_last优化
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    if config.get('channels_last', False):
                        inputs = inputs.to(memory_format=torch.channels_last)
                    outputs = model(inputs)
                    outputs = ensure_2d_logits(outputs)
                    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            else:
                if config.get('channels_last', False):
                    inputs = inputs.to(memory_format=torch.channels_last)
                outputs = model(inputs)
                outputs = ensure_2d_logits(outputs)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            
            test_preds_noisy.extend(probs)
            test_labels_noisy.extend(labels.cpu().numpy())
    
    test_noisy_time = time.time() - test_noisy_start_time
    test_auc_noisy = roc_auc_score(test_labels_noisy, test_preds_noisy)
    test_acc_noisy = accuracy_score(test_labels_noisy, (np.array(test_preds_noisy) > 0.5).astype(int))
    test_ap_noisy = average_precision_score(test_labels_noisy, test_preds_noisy)
    test_eer_noisy, test_eer_threshold_noisy = calculate_eer(test_labels_noisy, test_preds_noisy)
    
    logger.info(f"噪声测试结果:")
    logger.info(f"  - 测试AUC (噪声): {test_auc_noisy:.4f}")
    logger.info(f"  - 测试AP (噪声): {test_ap_noisy:.4f}")
    logger.info(f"  - 测试EER (噪声): {test_eer_noisy:.4f} (阈值: {test_eer_threshold_noisy:.4f})")
    logger.info(f"  - 测试准确率 (噪声): {test_acc_noisy:.4f}")
    logger.info(f"  - 测试时间: {test_noisy_time:.2f}秒")
    logger.info(f"  - 鲁棒性评估 (AUC下降): {test_auc - test_auc_noisy:.4f}")
    logger.info(f"  - 鲁棒性评估 (AP下降): {test_ap - test_ap_noisy:.4f}")
    logger.info(f"  - 鲁棒性评估 (EER上升): {test_eer_noisy - test_eer:.4f}")
    
    print(f"Test AUC (Noisy): {test_auc_noisy:.4f}, Test AP (Noisy): {test_ap_noisy:.4f}, Test EER (Noisy): {test_eer_noisy:.4f}, Test Acc (Noisy): {test_acc_noisy:.4f}")
    
    # 程序结束统计
    total_program_time = time.time() - program_start_time
    logger.info("\n" + "="*80)
    logger.info("程序执行完成 - 完整训练报告")
    logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"总执行时间: {total_program_time/3600:.2f} 小时")
    logger.info("="*80)
    
    # 训练配置回顾
    logger.info("训练配置回顾:")
    logger.info(f"  模型: DeiT3 Base (89M参数)")
    logger.info(f"  硬件: 4090 24GB")
    logger.info(f"  批次大小: {config['batch_size']} (有效: {config['batch_size'] * config['gradient_accumulation_steps']})")
    logger.info(f"  学习率: {config['learning_rate']} (warmup: {config['warmup_epochs']}轮)")
    logger.info(f"  训练轮数: {config['num_epochs']} (patience: {config['patience']})")
    logger.info(f"  优化策略: {'混合精度' if config['mixed_precision'] else '标准精度'} + {'层级学习率衰减' if config.get('use_layer_decay', False) else '统一学习率'}")
    logger.info(f"  正则化: Dropout={config['dropout_rate']}, 标签平滑={config['label_smoothing']}")
    logger.info(f"  内存优化: {'channels_last' if config.get('channels_last', False) else '标准格式'} + {'模型编译' if config.get('compile_model', False) else '无编译'}")
    
    # 性能指标总结
    logger.info("\n性能指标总结:")
    logger.info(f"  - 最佳验证AUC: {best_auc:.4f}")
    logger.info(f"  - 纯净测试AUC: {test_auc:.4f}")
    logger.info(f"  - 纯净测试AP: {test_ap:.4f}")
    logger.info(f"  - 纯净测试EER: {test_eer:.4f}")
    logger.info(f"  - 噪声测试AUC: {test_auc_noisy:.4f}")
    logger.info(f"  - 噪声测试AP: {test_ap_noisy:.4f}")
    logger.info(f"  - 噪声测试EER: {test_eer_noisy:.4f}")
    
    # 鲁棒性分析
    logger.info("\n鲁棒性分析:")
    logger.info(f"  - AUC保持率: {((test_auc_noisy/test_auc)*100):.1f}% (下降{((1-test_auc_noisy/test_auc)*100):.1f}%)")
    logger.info(f"  - AP保持率: {((test_ap_noisy/test_ap)*100):.1f}% (下降{((1-test_ap_noisy/test_ap)*100):.1f}%)")
    logger.info(f"  - EER变化: {test_eer:.4f} → {test_eer_noisy:.4f} (上升{((test_eer_noisy/test_eer-1)*100):+.1f}%)")
    
    # 训练效率分析
    estimated_samples_per_hour = (len(train_loader.dataset) + len(val_loader.dataset)) * config['num_epochs'] / (total_program_time / 3600)
    logger.info("\n训练效率分析:")
    logger.info(f"  - 训练样本总数: {len(train_loader.dataset):,}")
    logger.info(f"  - 验证样本总数: {len(val_loader.dataset):,}")
    logger.info(f"  - 测试样本总数: {len(test_loader.dataset):,}")
    logger.info(f"  - 处理速度: {estimated_samples_per_hour:.0f} 样本/小时")
    
    # GPU内存使用总结
    if torch.cuda.is_available():
        final_memory = torch.cuda.memory_allocated() / 1024**3
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        logger.info(f"  - 最终GPU内存: {final_memory:.2f}GB")
        logger.info(f"  - 峰值GPU内存: {peak_memory:.2f}GB")
        logger.info(f"  - 内存利用率: {(peak_memory/32)*100:.1f}% (V100 32GB)")
    
    # 文件路径总结
    logger.info("\n重要文件路径:")
    logger.info(f"  - 最佳模型: {best_model_path}")
    logger.info(f"  - 训练日志: {log_file_path}")
    logger.info(f"  - 配置保存: {config['model_save_dir']}")
    
    logger.info("="*80)
    logger.info("训练任务完成！所有配置和结果已完整记录到日志文件中。")
    logger.info("="*80)