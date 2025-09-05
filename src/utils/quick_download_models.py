#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的预训练模型下载脚本
下载指定的模型到本地，便于传输到远程服务器

Usage:
    # 列出支持的模型
    python quick_download_models.py --list

    # 下载模型并指定保存目录（Linux 路径示例）
    python quick_download_models.py --model swinv2_small_window16_256 --save-dir /home/user/models

    # 下载模型并指定保存目录（Windows 路径示例）
    python quick_download_models.py --model swinv2_small_window16_256 --save-dir "D:\\models\\pretrained"
"""

import os
import sys
import argparse
from pathlib import Path

def download_model(model_name, save_dir):
    """下载指定的预训练模型"""
    
    # 常用的模型列表
    SUPPORTED_MODELS = {
        # Swin Transformer V2 系列
        'swinv2_small_window16_256': 'Swin Transformer V2 Small 16x16 (50M params)',
        'swinv2_base_window16_256': 'Swin Transformer V2 Base 16x16 (88M params)', 
        'swinv2_tiny_window16_256': 'Swin Transformer V2 Tiny 16x16 (28M params)',
        
        # Swin Transformer V1 系列
        'swin_tiny_patch4_window7_224': 'Swin Transformer Tiny (28M params)',
        'swin_small_patch4_window7_224': 'Swin Transformer Small (50M params)',
        'swin_base_patch4_window7_224': 'Swin Transformer Base (88M params)',
        
        # Vision Transformer (最新版本)
        'vit_tiny_patch16_224': 'Vision Transformer Tiny (5.7M params)',
        'vit_small_patch16_224': 'Vision Transformer Small (22M params)',
        'vit_base_patch16_224': 'Vision Transformer Base (86M params)',
        'vit_large_patch16_224': 'Vision Transformer Large (304M params)',
        'deit3_small_patch16_224': 'DeiT v3 Small (22M params)',
        'deit3_base_patch16_224': 'DeiT v3 Base (86M params)',
        
        # ConvNeXt V2 系列
        'convnext_tiny': 'ConvNeXt Tiny (28M params)',
        'convnext_small': 'ConvNeXt Small (50M params)',
        'convnext_base': 'ConvNeXt Base (89M params)',
        'convnextv2_nano': 'ConvNeXt V2 Nano (15M params)',
        'convnextv2_tiny': 'ConvNeXt V2 Tiny (28M params)',
        'convnextv2_base': 'ConvNeXt V2 Base (89M params)',
        
        # EfficientNet V2 系列
        'efficientnetv2_s': 'EfficientNet V2 Small (21M params)',
        'efficientnetv2_m': 'EfficientNet V2 Medium (54M params)',
        'efficientnetv2_l': 'EfficientNet V2 Large (119M params)',
        'tf_efficientnetv2_s': 'EfficientNet V2 Small TF (21M params)',
        'tf_efficientnetv2_m': 'EfficientNet V2 Medium TF (54M params)',
        
        # MobileNet V3/V4 系列
        'mobilenetv3_large_100': 'MobileNet V3 Large (5.5M params)',
        'mobilenetv3_small_100': 'MobileNet V3 Small (2.5M params)',
        'mobilenetv4_conv_small': 'MobileNet V4 Conv Small (3.8M params)',
        'mobilenetv4_conv_medium': 'MobileNet V4 Conv Medium (9.4M params)',
        'mobilenetv4_conv_large': 'MobileNet V4 Conv Large (32M params)',
        
        # MobileViT 系列
        'mobilevit_s': 'MobileViT Small (5.6M params)',
        'mobilevit_xs': 'MobileViT XSmall (2.3M params)',
        'mobilevitv2_050': 'MobileViT v2 0.5x (1.4M params)',
        'mobilevitv2_075': 'MobileViT v2 0.75x (2.9M params)',
        'mobilevitv2_100': 'MobileViT v2 1.0x (4.9M params)',
    }
    
    if model_name not in SUPPORTED_MODELS:
        print(f"错误: 不支持的模型 '{model_name}'")
        print("\n支持的模型列表:")
        for i, (name, desc) in enumerate(SUPPORTED_MODELS.items(), 1):
            print(f"  {i:2d}. {name:<30} - {desc}")
        return False
    
    try:
        import torch
        import timm
    except ImportError as e:
        print(f"错误: 缺少必要的依赖包 - {e}")
        print("请运行: pip install torch timm")
        return False
    
    # 解析并创建保存目录，兼容 Windows/Linux 路径
    save_base = Path(save_dir).expanduser().resolve()
    save_path = save_base / model_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    model_file = save_path / f"{model_name}.pth"
    
    # 检查模型是否已存在
    if model_file.exists():
        print(f"模型已存在: {model_file}")
        file_size = model_file.stat().st_size / 1024 / 1024
        print(f"文件大小: {file_size:.1f} MB")
        return True
    
    print(f"开始下载模型: {model_name}")
    print(f"描述: {SUPPORTED_MODELS[model_name]}")
    
    try:
        # 创建模型并下载预训练权重
        print("正在下载预训练权重...")
        model = timm.create_model(model_name, pretrained=True)
        
        # 保存模型权重
        print(f"保存模型到: {model_file}")
        torch.save(model.state_dict(), model_file)
        
        # 保存模型信息
        info_file = save_path / "model_info.txt"
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Description: {SUPPORTED_MODELS[model_name]}\n")
            f.write(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
            f.write(f"Input Size: {getattr(model, 'default_cfg', {}).get('input_size', 'Unknown')}\n")
        
        file_size = model_file.stat().st_size / 1024 / 1024
        print(f"✓ 下载完成!")
        print(f"  保存位置: {model_file}")  
        print(f"  文件大小: {file_size:.1f} MB")
        print(f"  模型信息: {info_file}")
        
        return True
        
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        return False

def list_models():
    """列出支持的模型"""
    models = {
        # Swin Transformer V2 系列
        'swinv2_small_window16_256': 'Swin Transformer V2 Small 16x16 (50M params)',
        'swinv2_base_window16_256': 'Swin Transformer V2 Base 16x16 (88M params)', 
        'swinv2_tiny_window16_256': 'Swin Transformer V2 Tiny 16x16 (28M params)',
        
        # Swin Transformer V1 系列
        'swin_tiny_patch4_window7_224': 'Swin Transformer Tiny (28M params)',
        'swin_small_patch4_window7_224': 'Swin Transformer Small (50M params)',
        'swin_base_patch4_window7_224': 'Swin Transformer Base (88M params)',
        
        # Vision Transformer (最新版本)
        'vit_tiny_patch16_224': 'Vision Transformer Tiny (5.7M params)',
        'vit_small_patch16_224': 'Vision Transformer Small (22M params)',
        'vit_base_patch16_224': 'Vision Transformer Base (86M params)',
        'vit_large_patch16_224': 'Vision Transformer Large (304M params)',
        'deit3_small_patch16_224': 'DeiT v3 Small (22M params)',
        'deit3_base_patch16_224': 'DeiT v3 Base (86M params)',
        
        # ConvNeXt V2 系列
        'convnext_tiny': 'ConvNeXt Tiny (28M params)',
        'convnext_small': 'ConvNeXt Small (50M params)',
        'convnext_base': 'ConvNeXt Base (89M params)',
        'convnextv2_nano': 'ConvNeXt V2 Nano (15M params)',
        'convnextv2_tiny': 'ConvNeXt V2 Tiny (28M params)',
        'convnextv2_base': 'ConvNeXt V2 Base (89M params)',
        
        # EfficientNet V2 系列
        'efficientnetv2_s': 'EfficientNet V2 Small (21M params)',
        'efficientnetv2_m': 'EfficientNet V2 Medium (54M params)',
        'efficientnetv2_l': 'EfficientNet V2 Large (119M params)',
        'tf_efficientnetv2_s': 'EfficientNet V2 Small TF (21M params)',
        'tf_efficientnetv2_m': 'EfficientNet V2 Medium TF (54M params)',
        
        # MobileNet V3/V4 系列
        'mobilenetv3_large_100': 'MobileNet V3 Large (5.5M params)',
        'mobilenetv3_small_100': 'MobileNet V3 Small (2.5M params)',
        'mobilenetv4_conv_small': 'MobileNet V4 Conv Small (3.8M params)',
        'mobilenetv4_conv_medium': 'MobileNet V4 Conv Medium (9.4M params)',
        'mobilenetv4_conv_large': 'MobileNet V4 Conv Large (32M params)',
        
        # MobileViT 系列
        'mobilevit_s': 'MobileViT Small (5.6M params)',
        'mobilevit_xs': 'MobileViT XSmall (2.3M params)',
        'mobilevitv2_050': 'MobileViT v2 0.5x (1.4M params)',
        'mobilevitv2_075': 'MobileViT v2 0.75x (2.9M params)',
        'mobilevitv2_100': 'MobileViT v2 1.0x (4.9M params)',
    }
    
    print("=== 支持的预训练模型列表 ===")
    for i, (name, desc) in enumerate(models.items(), 1):
        print(f"{i:2d}. {name:<30} - {desc}")

def main():
    parser = argparse.ArgumentParser(description="简单的预训练模型下载工具")
    parser.add_argument('--model', type=str, required=True, help='要下载的模型名称')
    parser.add_argument('--save-dir', type=str, default='./pretrained_models',
                        help='模型保存目录，支持 Windows 或 Linux 路径 (默认: ./pretrained_models)')
    parser.add_argument('--list', action='store_true', help='列出支持的模型')
    
    args = parser.parse_args()
    
    # 列出模型
    if args.list:
        list_models()
        return

    # 下载指定模型
    success = download_model(args.model, args.save_dir)
    if not success:
        sys.exit(1)
    
    # 如果没有参数，显示帮助
    print("用法示例:")
    print("  python quick_download_models.py --list                              # 列出支持的模型")
    print("  python quick_download_models.py --model swinv2_small_window16_256   # 下载指定模型")
    print("  python quick_download_models.py --model swinv2_base_window16_256    # 下载Base版本")
    print("python quick_download_models.py --model swinv2_base_window16_256 --save-dir /home/user/models  # 指定保存目录 (Linux)")
    print("python quick_download_models.py --model swinv2_base_window16_256 --save-dir 'D:\models\pretrained' # 指定保存目录 (Windows)")

if __name__ == "__main__":
    main()
