import timm

# 测试Swin Transformer V2 Base模型加载
print("测试Swin Transformer V2 Base模型加载...")
try:
    model = timm.create_model('swinv2_base_window16_256', pretrained=True, num_classes=2)
    print("✓ 模型加载成功!")
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数数量: {total_params:,}")
    print(f"输入尺寸: {model.default_cfg.get('input_size', 'Unknown')}")
    print(f"模型架构: {model.__class__.__name__}")
    
    # 测试前向传播
    import torch
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(x)
        print(f"输出形状: {output.shape}")
    
    print("✓ 模型测试完成!")
    
except Exception as e:
    print(f"✗ 模型加载失败: {e}")
