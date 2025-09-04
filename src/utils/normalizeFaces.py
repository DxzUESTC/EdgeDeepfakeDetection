# -*- coding: utf-8 -*-
# @Author  : dxzzZ

import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 输入和输出路径（保持目录结构）
input_dir = "D:\\09_Project\\EdgeDeepfakeDetection\\data\\preprocessed\\faces"
output_dir = "D:\\09_Project\\EdgeDeepfakeDetection\\data\\preprocessed\\faces_256"
target_size = (256, 256)

def get_folder_prefix(path):
    """根据文件夹路径返回对应的前缀"""
    path_lower = path.lower()
    if 'original' in path_lower:
        return '0'
    elif 'deepfakes' in path_lower:
        return '1'
    elif 'face2face' in path_lower:
        return '2'
    elif 'faceshifter' in path_lower:
        return '3'
    elif 'faceswap' in path_lower:
        return '4'
    elif 'neuraltexture' in path_lower:
        return '5'
    else:
        return ''  # 如果没有匹配的文件夹，不添加前缀

def process_image(in_path, out_path):
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        # 获取前缀并修改输出文件名
        prefix = get_folder_prefix(in_path)
        if prefix:
            out_dir = os.path.dirname(out_path)
            filename = os.path.basename(out_path)
            new_filename = f"{prefix}{filename}"
            out_path = os.path.join(out_dir, new_filename)
        
        img = Image.open(in_path).convert("RGB")
        img = img.resize(target_size, Image.LANCZOS)
        img.save(out_path, "JPEG")
    except Exception as e:
        return f"Failed: {in_path}, Error: {e}"
    return None

def get_all_images(input_dir, output_dir):
    tasks = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                in_path = os.path.join(root, file)
                rel_path = os.path.relpath(in_path, input_dir)
                out_path = os.path.join(output_dir, rel_path)
                tasks.append((in_path, out_path))
    return tasks

def main():
    tasks = get_all_images(input_dir, output_dir)
    errors = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_image, in_path, out_path): in_path for in_path, out_path in tasks}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Resizing"):
            err = f.result()
            if err:
                errors.append(err)
    print("Done!")
    if errors:
        print("Some errors occurred:")
        for e in errors:
            print(e)

if __name__ == "__main__":
    main()
