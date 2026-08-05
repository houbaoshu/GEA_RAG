import os
import base64
from pathlib import Path
from typing import Optional, List
import torch

from src.config import config, AppConfig
from src.schema import (
    DocumentPage,
    Evidence,
    RetrievalOutput,
    QAAnalysisResult,
)
from src.utils.device import get_device_and_dtype


INDUSTRIAL_SYSTEM_PROMPT = """你是一个专业的工业文档与图纸分析助手，擅长理解工业产品手册、技术规格书、安装指南、CAD图纸和参数表。

分析要求：
1. 仔细阅读提供的文档页面图片，严格根据图片中的视觉信息（文本、表格数值、尺寸标注、部件标签等）回答用户问题。
2. 对于表格数据，请准确对应行列指标与数值；对于图纸或尺寸参数，请指明具体的尺寸数字及单位（如 mm, m³/h, kW, bar 等）。
3. 如果你在图片中定位到了关键回答区域（如特定图纸框、表格或说明文字），请在回答中附带坐标框定位，格式为：[ymin, xmin, ymax, xmax]（0-1000 归一化比例）。
4. 请在回答最后简要说明引用的关键页码与对应区域类型（如：参数表、安装尺寸图等）。
5. 如果提供的图片中找不到问题的答案，请明确回答：“提供的文档页面中未查找到相关信息。”
"""


class QwenVLAnalyzer:
    """
    Vision Language Understanding Engine based on Qwen-VL (ARCHITECTURE.md Section 6).
    Supports local HuggingFace Qwen2.5-VL transformers and OpenAI-compatible API modes.
    """

    def __init__(
        self,
        app_config: AppConfig = config,
        model_name: Optional[str] = None,
        backend: Optional[str] = None,
    ):
        self.cfg = app_config
        self.backend = backend or self.cfg.qwen_vl_backend
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = (
                self.cfg.qwen_vl_api_model_name
                if self.backend == "api"
                else self.cfg.qwen_vl_model_name
            )
        self.device, self.dtype = get_device_and_dtype()

        self.model = None
        self.processor = None
        self.api_client = None

    def _ensure_local_model(self):
        if self.model is None or self.processor is None:
            from src.utils.model_loader import load_qwen_vl_model

            print(f"🍏 初始化本地 Qwen-VL 视觉理解模型 ({self.model_name})...")
            self.processor, self.model = load_qwen_vl_model(
                self.model_name, self.device, self.dtype
            )

    def _ensure_api_client(self):
        if self.api_client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("使用 API 模式请确保已安装 openai 库: pip install openai")

            api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "❌ 未找到 API Key！请在 .env 中或环境变量中设置 DASHSCOPE_API_KEY 或 QWEN_API_KEY。"
                )

            base_url = os.getenv(
                "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            self.api_client = OpenAI(api_key=api_key, base_url=base_url)

    def analyze(
        self,
        query: str,
        retrieval_output: RetrievalOutput,
        system_prompt: Optional[str] = None,
    ) -> QAAnalysisResult:
        """
        根据检索到的 Top-K 页面图片和用户提问，调用 Qwen-VL 进行视觉理解与回答生成，
        并通过 EvidenceManager 构建持久化视觉证据对象。
        """
        from src.evidence.evidence_manager import EvidenceManager

        system_prompt = system_prompt or INDUSTRIAL_SYSTEM_PROMPT

        if not retrieval_output.results:
            return QAAnalysisResult(
                answer="提供的文档页面中未查找到相关信息。",
                evidence_page=None,
                evidence_region=None,
                evidences=[],
                raw_response="Empty retrieval results.",
            )

        # 提取相关图片路径
        image_paths = [item.page.image_path for item in retrieval_output.results]
        primary_item = retrieval_output.results[0]
        evidence_page = primary_item.page.page_number

        if self.backend == "local":
            raw_answer = self._analyze_local(query, image_paths, system_prompt)
        else:
            raw_answer = self._analyze_api(query, image_paths, system_prompt)

        # 调用 EvidenceManager 解析坐标并生成裁剪图/红框标注图
        evidence_mgr = EvidenceManager(self.cfg)
        result_obj = QAAnalysisResult(
            answer=raw_answer,
            evidence_page=evidence_page,
            evidence_region="Top-K Relevant Pages Context",
            evidences=[],
            raw_response=raw_answer,
        )
        evidences = evidence_mgr.process_rag_evidence(retrieval_output, result_obj)
        result_obj.evidences = evidences

        return result_obj

    def _analyze_local(
        self, query: str, image_paths: List[str], system_prompt: str
    ) -> str:
        """本地 HuggingFace Qwen2.5-VL 模态推理"""
        self._ensure_local_model()
        from qwen_vl_utils import process_vision_info

        content = []
        for img_path in image_paths:
            content.append({"type": "image", "image": f"file://{Path(img_path).resolve()}"})
        content.append({"type": "text", "text": query})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        return output_text[0].strip()

    def _analyze_api(
        self, query: str, image_paths: List[str], system_prompt: str
    ) -> str:
        """API 兼容模式多模态推理"""
        self._ensure_api_client()

        content = []
        for img_path in image_paths:
            with open(img_path, "rb") as f:
                b64_str = base64.b64encode(f.read()).decode("utf-8")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_str}"},
                }
            )
        content.append({"type": "text", "text": query})

        response = self.api_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
