import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(filename='train_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# 自定义数据集
class FFDataset(Dataset):
    def __init__(self, csv_file, faces_dir, split, transform=None):
        self.faces_dir = faces_dir
        self.transform = transform
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['split'] == split]
        self.image_paths = []
        self.labels = []
        
        type_map = {
            '0': 'original',
            '1': 'deepfakes',
            '2': 'Face2Face',
            '3': 'FaceShifter',
            '4': 'FaceSwap',
            '5': 'NeuralTexture'
        }
        
        for _, row in self.df.iterrows():
            video_id = row['video_id']
            label = row['label']
            type_id = video_id[0]
            video_type = type_map[type_id]
            face_dir = os.path.join(faces_dir, video_type, 'c23')
            if video_type == 'original':
                prefix = f"{video_id}_frame_"
            else:
                prefix = f"{video_id[:3]}_"
            for img_name in os.listdir(face_dir):
                if img_name.startswith(prefix):
                    self.image_paths.append(os.path.join(face_dir, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = add_h264_noise(image, quality=np.random.randint(20, 50))
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

# 数据加载（替换为实际路径）
train_dataset = FFDataset('/home/dxzzZ/Projects/EdgeDeepfakeDetection/data/datasets/metadata/video_metadata_c23.csv', 
                          '/home/dxzzZ/Projects/EdgeDeepfakeDetection/data/preprocessed/faces_224', 
                          split=0, transform=data_transforms['train'])
val_dataset = FFDataset('/home/dxzzZ/Projects/EdgeDeepfakeDetection/data/datasets/metadata/video_metadata_c23.csv',
                         '/home/dxzzZ/Projects/EdgeDeepfakeDetection/data/preprocessed/faces_224',
                           split=1, transform=data_transforms['val'])
test_dataset = FFDataset('/home/dxzzZ/Projects/EdgeDeepfakeDetection/data/datasets/metadata/video_metadata_c23.csv', 
                         '/home/dxzzZ/Projects/EdgeDeepfakeDetection/data/preprocessed/faces_224', 
                         split=2, transform=data_transforms['val'])
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)  # num_workers=8 利用多核
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8)

# 加载官方 MobileNetV3-Small
model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 多 GPU 支持
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    logging.info(f"Using {torch.cuda.device_count()} GPUs")

# 损失和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 训练函数（添加早停、checkpoint）
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, patience=5):
    best_auc = 0.0
    no_improve = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        logging.info(f"Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}")

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                val_preds.extend(probs)
                val_labels.extend(labels.cpu().numpy())
        val_auc = roc_auc_score(val_labels, val_preds)
        logging.info(f"Epoch {epoch+1}, Val AUC: {val_auc:.4f}")
        print(f"Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), 'mobilenetv3_best.pth')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Checkpoint 每 5 epoch 保存
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'mobilenetv3_epoch_{epoch+1}.pth')

        scheduler.step()
    return best_auc

# 训练
best_auc = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)
print(f"Best Validation AUC: {best_auc:.4f}")
logging.info(f"Best Validation AUC: {best_auc:.4f}")

# 测试
model.load_state_dict(torch.load('mobilenetv3_best.pth'))
model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        test_preds.extend(probs)
        test_labels.extend(labels.cpu().numpy())
test_auc = roc_auc_score(test_labels, test_preds)
print(f"Test AUC: {test_auc:.4f}")
logging.info(f"Test AUC: {test_auc:.4f}")