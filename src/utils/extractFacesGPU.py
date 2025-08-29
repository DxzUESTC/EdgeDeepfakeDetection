import os
import cv2
import numpy np
from retinaface.pre_trained_models import get_model
import torch

"""
这个脚本通过使用RetinaFace库来提取图像中的人脸，并将提取到的人脸（含外扩边界）保存为单独的图像文件。
"""

def get_device():
    """确定可用的最佳设备"""
    if torch.cuda.is_available():
        return "cuda"
    # elif torch.backends.mps.is_available():  # 对于Mac M1/M2
    #     return "mps"
    return "cpu"

def extract_faces(image_path, output_dir, padding=10, device=None):
    if device is None:
        device = get_device()
    
    # 初始化RetinaFace模型
    model = get_model("resnet50_2020-07-20", max_size=2048)
    model.eval()
    if device == "cuda":
        model = model.to("cuda")
    
    # 读取原始图像
    if device == "cuda":
        # 使用CUDA加速的图像读取
        stream = cv2.cuda_Stream()
        img_gpu = cv2.cuda.imread(image_path)
        img = img_gpu.download()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w = img.shape[:2]
    
    # 检测人脸位置
    faces = model.predict_jsons(img_rgb)
    
    if isinstance(faces, list):
        # 获取原始文件名(不含扩展名)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 找出最大面积的人脸
        max_area = 0
        max_face = None
        
        for face_data in faces:
            facial_area = face_data['bbox']
            # 确保坐标值为整数
            x1, y1, x2, y2 = map(int, facial_area)
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                max_face = face_data
        
        if max_face:
            # 获取最大人脸的坐标并转换为整数
            x1, y1, x2, y2 = map(int, max_face['bbox'])
            
            # 扩展边界（加入padding）
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # 裁剪人脸区域
            face_img = img_rgb[int(y1):int(y2), int(x1):int(x2)]
            
            # 生成输出文件名
            output_file = os.path.join(output_dir, f"{base_name}_face.jpg")
            # 保存人脸图像
            cv2.imwrite(output_file, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
            print(f"Saved face image to {output_file}")

# 批量提取路径下的所有图像中的人脸
def extract_faces_from_directory(directory, output_dir, padding=10):
    device = get_device()
    print(f"Using device: {device}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)
            try:
                extract_faces(image_path, output_dir, padding, device)
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")

if __name__ == "__main__":
    directory = input("Enter the input directory of images: ") 
    output_dir = input("Enter the output directory for extracted faces: ")
    padding = int(input("Enter the padding size for face extraction (default is 10): ") or 10)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    extract_faces_from_directory(directory, output_dir, padding)
    print("Face extraction completed.")
