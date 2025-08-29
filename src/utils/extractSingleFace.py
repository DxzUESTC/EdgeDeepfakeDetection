import logging
from pathlib import Path
import cv2
import numpy as np
from retinaface import RetinaFace
from typing import List, Optional, Union

def clean_path(path_str: str) -> str:
    """清理路径字符串，移除引号和多余的空格"""
    return path_str.strip().strip('"\'')

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def extract_faces(image_path: Union[str, Path], 
                 output_dir: Union[str, Path], 
                 padding: int = 10) -> List[Path]:
    """
    从图像中提取所有人脸并保存
    
    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
        padding: 人脸区域扩展像素值
        
    Returns:
        List[Path]: 保存的人脸图像文件路径列表
    """
    # 清理并转换为Path对象
    image_path = Path(clean_path(str(image_path)))
    output_dir = Path(clean_path(str(output_dir)))
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取原始图像
    img = cv2.imread(str(image_path))
    if img is None:
        logging.error(f"无法读取图像: {image_path}")
        return []
    
    h, w = img.shape[:2]
    saved_faces: List[Path] = []
    
    try:
        # 检测人脸位置
        faces = RetinaFace.detect_faces(img_path=str(image_path))
    except Exception as e:
        logging.error(f"人脸检测失败: {str(e)}")
        return []
    
    if not isinstance(faces, dict):
        logging.warning(f"未检测到人脸: {image_path}")
        return []
        
    # 获取原始文件名(不含扩展名)
    base_name = image_path.stem
    
    # 处理所有检测到的人脸
    for i, (face_key, face_data) in enumerate(faces.items()):
        try:
            # 获取人脸坐标
            x1, y1, x2, y2 = face_data['facial_area']
            
            # 扩展边界（加入padding）
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # 裁剪人脸区域
            face_img = img[y1:y2, x1:x2]
            
            # 生成输出文件路径
            output_file = output_dir / f"{base_name}_face_{i+1}.jpg"
            
            # 保存人脸图像
            cv2.imwrite(str(output_file), face_img)
            saved_faces.append(output_file)
            logging.info(f"已保存人脸图像: {output_file}")
            
        except Exception as e:
            logging.error(f"处理人脸 {i+1} 时出错: {str(e)}")
            continue
    
    return saved_faces

if __name__ == "__main__":
    try:
        # 获取输入并清理路径
        image_path = clean_path(input("请输入图像文件路径: "))
        # 固定输出目录
        output_dir = "D:/09_Project/BWTAC2025/EdgeDeepfakeDetection/experiments/temp"
        
        # 使用默认padding值10
        saved_faces = extract_faces(image_path, output_dir)
        logging.info(f"\n处理完成,共保存了 {len(saved_faces)} 个人脸图像.")
        
    except KeyboardInterrupt:
        logging.info("\n用户中断处理")
    except Exception as e:
        logging.error(f"程序执行出错: {str(e)}")