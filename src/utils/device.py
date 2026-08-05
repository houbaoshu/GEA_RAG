import torch


def get_device_and_dtype() -> tuple[str, torch.dtype]:
    """检测硬件加速支持并返回 (device, dtype)"""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"🚀 GPU 加速就绪: {gpu_name} (精度: {dtype})")
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
        print(f"🍏 Mac Apple Silicon 硬件加速就绪 (精度: {dtype})")
    else:
        device = "cpu"
        dtype = torch.float32
        print(f"⚠️ 未检测到 GPU，使用 CPU 运行 (精度: {dtype})")
    return device, dtype
