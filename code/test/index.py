import torch

print(f"PyTorch 版本: {torch.__version__}")

# 检查是否支持 CUDA
print(f"是否支持 CUDA: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # 打印 CUDA 版本
    print(f"CUDA 版本: {torch.version.cuda}")
    # 打印可用的 GPU 数量
    print(f"可用的 GPU 数量: {torch.cuda.device_count()}")
    # 打印当前 GPU 的名称
    print(f"当前 GPU 名称: {torch.cuda.get_device_name(0)}")
