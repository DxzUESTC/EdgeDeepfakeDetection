import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
import platform

# 解决一些常见的兼容性问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

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
    try:
        # 如果没有指定日志目录，使用当前脚本目录
        if log_dir is None:
            log_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        # 同时输出到控制台
        print(f"设置日志目录: {log_dir}")
        
        # 获取或创建logger
        logger = logging.getLogger('kd_train_logger')
        
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
        log_filename = os.path.join(log_dir, f'kd_train_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        
        # Windows兼容性：确保文件路径正确
        log_filename = os.path.normpath(log_filename)
        print(f"日志文件: {log_filename}")
        
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
    except Exception as e:
        print(f"设置日志系统时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
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
    返回知识蒸馏训练配置参数
    """
    config = {
        # 路径配置
        'base_dir': r'D:\09_Project\EdgeDeepfakeDetection',
        'log_dir': r'D:\09_Project\EdgeDeepfakeDetection\experiments\server',
        'model_save_dir': r'D:\09_Project\EdgeDeepfakeDetection\models\server\kd_mobilenetv3',
        'teacher_model_path': r'D:\09_Project\EdgeDeepfakeDetection\models\baseline\swin_transformer\seed42\swin_tiny_best.pth',
        
        # 随机种子配置
        'random_seed': 42,
        
        # 训练参数
        'batch_size': 64,  # 减小批次大小以同时加载两个模型
        'num_workers': 4,
        'num_epochs': 40,
        'patience': 8,
        'learning_rate': 5e-4,  # 降低学习率
        'weight_decay': 1e-4,
        
        # 知识蒸馏参数
        'temperature': 4.0,      # 蒸馏温度
        'alpha': 0.7,           # 蒸馏损失权重
        'beta': 0.3,            # 硬标签损失权重
        
        # 数据参数
        'add_train_noise': False,
        'add_test_noise': True,
    }
    return config

# 知识蒸馏损失函数
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, temperature=4.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, labels):
        # 硬标签损失 (与真实标签的交叉熵)
        hard_loss = self.criterion(student_logits, labels)
        
        # 软标签损失 (与教师模型输出的KL散度)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # 总损失
        total_loss = self.alpha * soft_loss + self.beta * hard_loss
        
        return total_loss, hard_loss, soft_loss

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
    
    def add_h264_noise(self, image):
        """添加H.264压缩噪声"""
        try:
            # 转换为uint8格式
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            # 随机选择压缩质量 (15-35, 数值越小压缩越厉害)
            quality = np.random.randint(15, 36)
            
            # H.264编码参数
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
            
            # 编码解码过程
            result, encoded_img = cv2.imencode('.jpg', image, encode_param)
            if result:
                decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
                return decoded_img.astype(np.float32) / 255.0
            else:
                return image.astype(np.float32) / 255.0
        except:
            return image.astype(np.float32) / 255.0
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 只在训练时添加噪声，测试时保持原始图像
        if hasattr(self, 'add_noise') and self.add_noise:
            image = self.add_h264_noise(image)
        if self.transform:
            # 转换为PIL Image
            from PIL import Image
            image = Image.fromarray(image)
            image = self.transform(image)
        return image, label

def load_teacher_model(model_path, device):
    """加载教师模型(SwinT)"""
    logger.info(f"正在加载教师模型: {model_path}")
    
    try:
        # 创建SwinT模型
        teacher_model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=2)
        
        # 加载预训练权重
        checkpoint = torch.load(model_path, map_location=device)
        
        # 检查是否需要处理state_dict格式
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            logger.info("检测到checkpoint包含state_dict键")
            checkpoint = checkpoint['state_dict']
        
        teacher_model.load_state_dict(checkpoint, strict=False)  # 使用strict=False允许不完全匹配
        teacher_model = teacher_model.to(device)
        teacher_model.eval()  # 教师模型设为评估模式
        
        # 冻结教师模型参数
        for param in teacher_model.parameters():
            param.requires_grad = False
        
        logger.info("✓ 教师模型加载完成")
        return teacher_model
        
    except Exception as e:
        logger.error(f"加载教师模型时出错: {str(e)}")
        raise

def create_student_model(device):
    """创建学生模型(MobileNetV3)"""
    logger.info("正在创建学生模型(MobileNetV3)...")
    
    # 加载预训练的MobileNetV3
    student_model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
    
    # 修改分类头
    student_model.classifier[3] = nn.Linear(student_model.classifier[3].in_features, 2)
    student_model = student_model.to(device)
    
    logger.info("✓ 学生模型创建完成")
    return student_model

def train_with_distillation(teacher_model, student_model, train_loader, val_loader, 
                           distill_loss_fn, optimizer, scheduler, config, save_dir):
    """使用知识蒸馏进行训练"""
    
    num_epochs = config['num_epochs']
    patience = config['patience']
    device = next(student_model.parameters()).device
    
    best_val_acc = 0.0
    best_val_auc = 0.0
    patience_counter = 0
    
    # 保存路径
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, 'kd_mobilenetv3_best.pth')
    
    logger.info("开始知识蒸馏训练...")
    logger.info(f"总轮数: {num_epochs}, 早停耐心: {patience}")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # 训练阶段
        student_model.train()
        teacher_model.eval()  # 教师模型始终保持评估模式
        
        train_loss = 0.0
        train_hard_loss = 0.0
        train_soft_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', 
                           leave=False, ncols=100)
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # 学生模型前向传播
            student_outputs = student_model(images)
            
            # 教师模型前向传播 (不计算梯度)
            with torch.no_grad():
                teacher_outputs = teacher_model(images)
            
            # 计算蒸馏损失
            total_loss, hard_loss, soft_loss = distill_loss_fn(student_outputs, teacher_outputs, labels)
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += total_loss.item()
            train_hard_loss += hard_loss.item()
            train_soft_loss += soft_loss.item()
            
            _, predicted = torch.max(student_outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # 更新进度条
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.1f}%'
                })
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        avg_hard_loss = train_hard_loss / len(train_loader)
        avg_soft_loss = train_soft_loss / len(train_loader)
        
        # 验证阶段
        student_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_targets = []
        val_probs = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', 
                               leave=False, ncols=100)
            
            for images, labels in progress_bar:
                images, labels = images.to(device), labels.to(device)
                
                student_outputs = student_model(images)
                teacher_outputs = teacher_model(images)
                
                total_loss, _, _ = distill_loss_fn(student_outputs, teacher_outputs, labels)
                val_loss += total_loss.item()
                
                _, predicted = torch.max(student_outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # 收集预测结果用于AUC计算
                probs = F.softmax(student_outputs, dim=1)
                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                val_probs.extend(probs[:, 1].cpu().numpy())  # 取正类概率
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # 计算AUC
        val_auc = roc_auc_score(val_targets, val_probs) if len(set(val_targets)) > 1 else 0.0
        
        # 更新学习率
        if scheduler:
            if hasattr(scheduler, 'step'):
                if hasattr(scheduler, 'get_last_lr'):
                    scheduler.step()
                else:
                    scheduler.step(avg_val_loss)
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 计算时间
        epoch_time = time.time() - start_time
        
        # 记录详细信息
        logger.info(f"Epoch {epoch+1}/{num_epochs}:")
        logger.info(f"  训练 - 损失: {avg_train_loss:.4f} (硬: {avg_hard_loss:.4f}, 软: {avg_soft_loss:.4f}), 准确率: {train_acc:.2f}%")
        logger.info(f"  验证 - 损失: {avg_val_loss:.4f}, 准确率: {val_acc:.2f}%, AUC: {val_auc:.4f}")
        logger.info(f"  学习率: {current_lr:.6f}, 时间: {epoch_time:.1f}s")
        
        # 保存最佳模型
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_auc > best_val_auc):
            best_val_acc = val_acc
            best_val_auc = val_auc
            patience_counter = 0
            
            torch.save(student_model.state_dict(), best_model_path)
            logger.info(f"  ✓ 新的最佳模型已保存! 验证准确率: {val_acc:.2f}%, AUC: {val_auc:.4f}")
        else:
            patience_counter += 1
            logger.info(f"  未改善 ({patience_counter}/{patience})")
        
        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f'kd_mobilenetv3_epoch_{epoch+1}.pth')
            checkpoint_path = os.path.normpath(checkpoint_path)  # Windows兼容性
            torch.save(student_model.state_dict(), checkpoint_path)
            logger.info(f"  检查点已保存: {checkpoint_path}")
        
        # 早停检查
        if patience_counter >= patience:
            logger.info(f"\n早停触发! 最佳验证准确率: {best_val_acc:.2f}%, AUC: {best_val_auc:.4f}")
            break
        
        logger.info("-" * 50)
    
    logger.info("知识蒸馏训练完成!")
    logger.info(f"最佳模型保存路径: {best_model_path}")
    logger.info(f"最佳验证准确率: {best_val_acc:.2f}%")
    logger.info(f"最佳验证AUC: {best_val_auc:.4f}")
    
    return best_model_path, best_val_acc, best_val_auc

def main():
    global logger, log_file_path
    
    # 获取配置
    config = get_config()
    
    # 设置随机种子
    if config.get('random_seed') is not None:
        set_random_seed(config['random_seed'])
        print(f"✓ 随机种子已设置为: {config['random_seed']}")
    else:
        print("⚠ 未设置随机种子，结果将是随机的")
    
    # 设置日志
    log_dir = config['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    logger, log_file_path = setup_detailed_logging(log_dir)
    
    logger.info("=" * 60)
    logger.info("知识蒸馏训练 - SwinT -> MobileNetV3")
    logger.info("=" * 60)
    
    # 记录配置信息
    logger.info("训练配置:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # 检查教师模型是否存在
    teacher_path = config['teacher_model_path']
    if not os.path.exists(teacher_path):
        logger.error(f"教师模型文件不存在: {teacher_path}")
        return
    
    # 数据预处理 (图片已经是224x224，无需resize)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 数据路径
    csv_file = os.path.join(config['base_dir'], 'data', 'datasets', 'metadata', 'video_metadata_c23.csv')
    faces_dir = os.path.join(config['base_dir'], 'data', 'preprocessed', 'faces_224')
    
    logger.info("正在加载数据集...")
    
    # 创建数据集
    train_dataset = FFDataset(
        csv_file=csv_file, 
        faces_dir=faces_dir, 
        split=0,  # 训练集使用split=0
        transform=train_transform,
        add_noise=config.get('add_train_noise', False)
    )
    
    val_dataset = FFDataset(
        csv_file=csv_file, 
        faces_dir=faces_dir, 
        split=1,  # 验证集使用split=1
        transform=val_transform,
        add_noise=False
    )
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    
    # 统计标签分布
    if len(train_dataset) > 0:
        train_labels = np.array(train_dataset.labels)
        logger.info(f"训练集标签分布 - 真实: {np.sum(train_labels == 0)}, 伪造: {np.sum(train_labels == 1)}")
    
    if len(val_dataset) > 0:
        val_labels = np.array(val_dataset.labels)
        logger.info(f"验证集标签分布 - 真实: {np.sum(val_labels == 0)}, 伪造: {np.sum(val_labels == 1)}")
    
    # 检查数据集是否为空
    if len(train_dataset) == 0:
        logger.error("训练集为空！请检查数据路径和元数据文件")
        return
    if len(val_dataset) == 0:
        logger.error("验证集为空！请检查数据路径和元数据文件")
        return
    
    # 创建数据加载器
    # 使用较少的worker数量，避免多进程问题
    num_workers = 0  # 暂时设为0，避免多进程问题
    logger.info(f"设置num_workers为: {num_workers} (避免多进程问题)")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=num_workers,
        worker_init_fn=None if num_workers == 0 else worker_init_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=num_workers,
        worker_init_fn=None if num_workers == 0 else worker_init_fn,
        pin_memory=True
    )
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    logger.info(f"批次大小: {config['batch_size']}")
    
    # 检查CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA设备信息: {torch.cuda.get_device_name()}")
        logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Windows CUDA兼容性优化
        if platform.system() == 'Windows':
            torch.backends.cudnn.benchmark = True
            logger.info("Windows CUDA优化已启用")
    else:
        logger.warning("未检测到CUDA，将使用CPU训练（可能会很慢）")
    
    # 加载教师模型
    teacher_model = load_teacher_model(teacher_path, device)
    
    # 创建学生模型
    student_model = create_student_model(device)
    
    # 创建蒸馏损失函数
    distill_loss_fn = DistillationLoss(
        alpha=config['alpha'], 
        beta=config['beta'], 
        temperature=config['temperature']
    )
    
    logger.info(f"蒸馏参数: α={config['alpha']}, β={config['beta']}, T={config['temperature']}")
    
    # 优化器和学习率调度器
    optimizer = optim.AdamW(
        student_model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    # 使用CosineAnnealingLR调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['num_epochs'], 
        eta_min=1e-6
    )
    
    # 开始知识蒸馏训练
    save_dir = config['model_save_dir']
    
    best_model_path, best_acc, best_auc = train_with_distillation(
        teacher_model=teacher_model,
        student_model=student_model,
        train_loader=train_loader,
        val_loader=val_loader,
        distill_loss_fn=distill_loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        save_dir=save_dir
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("知识蒸馏训练完成!")
    logger.info(f"最佳模型: {best_model_path}")
    logger.info(f"最佳准确率: {best_acc:.2f}%")
    logger.info(f"最佳AUC: {best_auc:.4f}")
    logger.info(f"日志文件: {log_file_path}")
    logger.info("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
