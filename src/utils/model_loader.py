import torch
from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def load_colqwen_model(
    model_name: str, device: str, dtype: torch.dtype
) -> tuple[ColQwen3_5Processor, ColQwen3_5]:
    """加载 ColQwen3.5 模型与 Processor"""
    print(f"\n📦 正在加载 ColQwen 模型: {model_name} ...")
    processor = ColQwen3_5Processor.from_pretrained(model_name)

    if device == "cuda":
        model = ColQwen3_5.from_pretrained(
            model_name, torch_dtype=dtype, device_map="cuda"
        ).eval()
    else:
        model = (
            ColQwen3_5.from_pretrained(model_name, torch_dtype=dtype)
            .to(device)
            .eval()
        )

    return processor, model


def load_qwen_vl_model(
    model_name: str, device: str, dtype: torch.dtype
) -> tuple[AutoProcessor, Qwen2_5_VLForConditionalGeneration]:
    """加载 Qwen2.5-VL 视觉语言大模型与 Processor"""
    print(f"\n📦 正在加载 Qwen-VL 视觉理解模型: {model_name} ...")
    processor = AutoProcessor.from_pretrained(model_name)

    if device == "cuda":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype, device_map="cuda"
        ).eval()
    else:
        model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=dtype
            )
            .to(device)
            .eval()
        )

    return processor, model

