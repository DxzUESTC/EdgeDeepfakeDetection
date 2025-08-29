# Edge Deepfake Detection



## 论文题目

Blockchain Enhanced Multi-Layer Edge Deepfake Detection: A framework for real-time deepfake detection of smartphone video calls.（暂定）





## **项目整体目标**

构建一个 **多层级的深伪检测系统**：

- **边缘服务器端**：采用性能较强的模型（Swin-T），通过 **联邦学习** 共享知识、提升检测精度。
- **移动端**：使用轻量化模型（ConvNeXt-Tiny，经知识蒸馏优化）实现实时推理。
- **区块链**：作为联邦学习与模型下发的安全保障机制，确保数据隐私、模型完整性和更新可追溯性。

------





## **主要工作**

1. **数据处理与建模**
   - 使用 FaceForensics++ 数据集作为核心训练集。
   - 服务器端模型：Swin-T，保证较强的表征能力。
   - 手机端模型：ConvNeXt-Tiny，通过蒸馏获得服务器模型的知识，提升轻量模型在真实场景下的表现。
2. **跨设备模型设计**
   - 边缘服务器运行大模型，支持复杂检测任务。
   - 移动端运行轻量模型，保证低延迟和实时性。
   - 两者通过蒸馏实现 **知识迁移**，保持模型体系的连贯性。
3. **训练与协作机制**
   - **联邦学习**：边缘服务器之间共享参数梯度，不暴露原始视频数据，提升检测效果。
   - **知识蒸馏**：大模型 → 小模型，保证端侧可用性。
4. **安全与可信**
   - 使用 **区块链技术** 记录联邦学习更新、模型版本和数据流转，增强系统的透明性与可溯源性。

------





## **创新点**

1. **跨层次模型协作**：提出“服务器大模型 + 手机小模型”的协同检测体系，兼顾精度与部署。
2. **双重机制融合**：将 **联邦学习（服务器间）** 与 **知识蒸馏（服务器到终端）** 结合，形成完整的知识流转路径。
3. **区块链保障**：不仅用于数据安全，也保证了模型更新过程的可追溯性和可信度。





## 数据集准备

FaceForensics++ （仅包含视频部分）

- [x] original：E:\DataSet\FaceForensics\original_sequences\youtube
  - [x] raw
  - [x] c23
  - [x] c40
- [x] Deepfakes：E:\DataSet\FaceForensics\manipulated_sequences\Deepfakes
  - [x] raw
  - [x] c23
  - [x] c40
- [x] Face2Face：E:\DataSet\FaceForensics\manipulated_sequences\Face2Face
  - [x] raw
  - [x] c23
  - [x] c40
- [x] FaceShifter：E:\DataSet\FaceForensics\manipulated_sequences\FaceShifter
  - [x] raw
  - [x] c23
  - [x] c40
- [ ] FaceSwap：E:\DataSet\FaceForensics\manipulated_sequences\FaceSwap
  - [ ] raw
  - [x] c23
  - [x] c40
- [ ] NeuralTextures：E:\DataSet\FaceForensics\manipulated_sequences\NeuralTextures
  - [ ] raw
  - [x] c23
  - [x] c40



## 数据的预处理

1. 数据存放与组织

   ```cmd
   dataset/
    ├── original/
    │    ├── raw/
    │    └── c23/
    ├── deepfakes/
    │    ├── raw/
    │    └── c23/
    └── metadata/
         └── video_index.csv
   ```

   `original/`：存放真实视频

   `deepfakes/`：存放伪造视频

   `metadata/`：存放视频索引表、标注信息

2. 建立索引表

   示意目录：

   | video\_id | path                           | label | split |
   | --------- | ------------------------------ | ----- | ----- |
   | 0001      | dataset/original/raw/0001.mp4  | real  | train |
   | 0002      | dataset/deepfakes/c23/0002.mp4 | fake  | train |
   | 0003      | dataset/original/raw/0003.mp4  | real  | val   |
   | ...       | ...                            | ...   | ...   |

   **video_id**：唯一标识符

   **path**：视频的存储路径

   **label**：`real/fake`

   **split**：train / val / test（随机划分 70/15/15）

3. 特征提取

   视频 → 模型输入

   (1) 抽帧（frame extraction）

   输出目录示意：

   ```cmd
   dataset_frames/
    ├── original/0001/0001_frame001.jpg
    ├── original/0001/0001_frame002.jpg
    ├── deepfakes/0002/0002_frame001.jpg
    └── ...
   ```

   (2) 人脸检测 + 对齐

   对每帧用 **RetinaFace** 检测人脸，截取并对齐。

   只保留人脸区域作为输入，因为深伪主要改动脸部特征。

   输出目录示意：

   ```cmd
   faces/
    ├── original/0001/0001_face001.jpg
    ├── deepfakes/0002/0002_face001.jpg
   ```

   (3) 人脸图像预处理

   - resize 到模型需要的分辨率（Swin-T / ConvNeXt 通常 224×224 或 384×384）。

   - normalize（ImageNet 均值方差）。





```
pip install torch==1.7.0+cu102 torchvision==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install pandas numpy opencv-python scikit-learn tqdm
```





train_dataset = FFDataset('/root/autodl-tmp/data/EDD/datasets/metadata/video_metadata_c23.csv', 

​             '/root/autodl-tmp/data/EDD/preprocessed/faces_224', 

​             split=0, transform=data_transforms['train'])

val_dataset = FFDataset('/root/autodl-tmp/data/EDD/datasets/metadata/video_metadata_c23.csv',

​             '/root/autodl-tmp/data/EDD/preprocessed/faces_224',

​              split=1, transform=data_transforms['val'])

test_dataset = FFDataset('/root/autodl-tmp/data/EDD/datasets/metadata/video_metadata_c23.csv', 

​             '/root/autodl-tmp/data/EDD/preprocessed/faces_224', 

​             split=2, transform=data_transforms['val'])
