import sys
print(f"Python版本: {sys.version}")

try:
    import torch
    print(f"✅ PyTorch版本: {torch.__version__}")
    print(f"✅ CUDA是否可用: {torch.cuda.is_available()}")
    print(f"✅ 安装成功！")
    
    # 简单测试
    x = torch.randn(2, 3)
    print(f"测试张量: {x}")
    
except ImportError as e:
    print(f"❌ 导入失败: {e}")
except Exception as e:
    print(f"❌ 运行时错误: {e}")