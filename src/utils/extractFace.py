import os
import cv2
import numpy as np
from retinaface import RetinaFace

"""
这个脚本通过使用RetinaFace库来提取图像中的人脸，并将提取到的人脸（含外扩边界）保存为单独的图像文件。
"""

def extract_faces(image_path, output_dir, padding=10):
    # 获取原始文件名(不含扩展名)并生成输出文件名
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}_face.jpg")
    
    # 检查文件是否已存在
    if os.path.exists(output_file):
        print(f"Skipping {output_file} - file already exists")
        return
    
    # 读取原始图像
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # 检测人脸位置
    faces = RetinaFace.detect_faces(img_path=image_path)
    
    if isinstance(faces, dict):
        # 找出最大面积的人脸
        max_area = 0
        max_face = None
        
        for face_key, face_data in faces.items():
            facial_area = face_data['facial_area']
            x1, y1, x2, y2 = facial_area
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                max_face = face_data
        
        if max_face:
            # 获取最大人脸的坐标
            x1, y1, x2, y2 = max_face['facial_area']
            
            # 扩展边界（加入padding）
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # 裁剪人脸区域
            face_img = img_rgb[y1:y2, x1:x2]
            
            # 保存人脸图像
            cv2.imwrite(output_file, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
            print(f"Saved face image to {output_file}")

# 批量提取路径下的所有图像中的人脸
def extract_faces_from_directory(directory, output_dir, padding=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)
            extract_faces(image_path, output_dir, padding)

if __name__ == "__main__":
    # 获取输入路径并处理可能的引号
    directory = input("Enter the input directory of images: ").strip('"')
    output_dir = input("Enter the output directory for extracted faces: ").strip('"')
    padding = int(input("Enter the padding size for face extraction (default is 10): ") or 10)
    
    # 验证路径格式
    directory = os.path.normpath(directory)
    output_dir = os.path.normpath(output_dir)
    
    # 检查输入路径是否存在
    if not os.path.exists(directory):
        print(f"Error: Input directory '{directory}' does not exist")
        exit(1)
        
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Processing images from: {directory}")
    print(f"Saving faces to: {output_dir}")
    
    extract_faces_from_directory(directory, output_dir, padding)
    print("Face extraction completed.")
