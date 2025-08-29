# -*- coding: utf-8 -*-
# @Author  : dxzzZ

import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 输入和输出路径（保持目录结构）
input_dir = "D:\\09_Project\\BWTAC2025\\EdgeDeepfakeDetection\\data\\preprocessed\\faces"
output_dir = "D:\\09_Project\\BWTAC2025\\EdgeDeepfakeDetection\\data\\preprocessed\\faces_224"
target_size = (224, 224)

def process_image(in_path, out_path):
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
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
