import re
import json
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image

from src.config import config, AppConfig
from src.schema import DocumentPage, Evidence, SearchResultItem, RetrievalOutput, QAAnalysisResult
from src.utils.image_utils import (
    normalize_bbox,
    crop_image_region,
    draw_bounding_box_overlay,
)


class EvidenceManager:
    """
    Evidence Manager Layer (ARCHITECTURE.md Section 7).
    Responsible for extracting page citations, parsing 2D bounding boxes,
    cropping regional evidence images, and generating annotated visual artifacts.
    """

    def __init__(self, app_config: AppConfig = config):
        self.cfg = app_config
        self.cfg.ensure_directories()
        self.evidence_dir = self.cfg.evidence_dir

    def parse_bboxes_from_text(self, text: str) -> List[Tuple[List[int | float], str]]:
        """
        从模型输出文本中提取 BoundingBox 坐标及说明。
        支持的匹配模式:
        1. Qwen 标签模式: <box>(ymin,xmin),(ymax,xmax)</box>
        2. JSON / 数组坐标模式: [ymin, xmin, ymax, xmax] 或 {"bbox": [...]}
        """
        results: List[Tuple[List[int | float], str]] = []

        # 模式 1: Qwen 标签 <box>(ymin,xmin),(ymax,xmax)</box>
        qwen_box_pattern = r"<box>\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)</box>"
        for match in re.finditer(qwen_box_pattern, text):
            ymin, xmin, ymax, xmax = map(int, match.groups())
            results.append(([ymin, xmin, ymax, xmax], "Target Region"))

        # 模式 2: 方括号坐标 [ymin, xmin, ymax, xmax]
        array_pattern = r"\[(\d{1,4}),\s*(\d{1,4}),\s*(\d{1,4}),\s*(\d{1,4})\]"
        for match in re.finditer(array_pattern, text):
            coords = list(map(int, match.groups()))
            # 过滤不是年份或普通列表的情况
            if any(v > 0 for v in coords):
                results.append((coords, "Parameter Region"))

        return results

    def create_evidence(
        self,
        document_id: str,
        page_number: int,
        image_path: str,
        bbox_raw: Optional[List[int | float]] = None,
        region_label: str = "Evidence Region",
    ) -> Evidence:
        """
        根据文档信息、图片路径和坐标构建完整的 Evidence 对象，
        并持久化保存裁剪图和红框标注图。
        """
        img_path = Path(image_path)
        doc_filename = f"{document_id}.pdf" if not document_id.endswith(".pdf") else document_id

        if not img_path.exists():
            return Evidence(
                document=doc_filename,
                page=page_number,
                image=str(image_path),
                region=region_label,
            )

        with Image.open(img_path) as img:
            img_w, img_h = img.size

        # 若未提取到有效坐标，则设置默认中心展示区域 (例如页面中间区域 [200, 100, 800, 900])
        if not bbox_raw:
            bbox_raw = [200, 100, 800, 900]

        bbox_pixels = normalize_bbox(bbox_raw, img_w, img_h)

        # 生成导出持久化文件名
        bbox_str = "_".join(map(str, bbox_pixels))
        crop_filename = f"crop_{document_id}_p{page_number}_{bbox_str}.png"
        annotated_filename = f"annotated_{document_id}_p{page_number}_{bbox_str}.png"

        cropped_out_path = self.evidence_dir / crop_filename
        annotated_out_path = self.evidence_dir / annotated_filename

        # 导出裁剪图与红框标注图
        crop_image_region(img_path, bbox_pixels, cropped_out_path)
        draw_bounding_box_overlay(
            img_path, bbox_pixels, annotated_out_path, label=region_label
        )

        return Evidence(
            document=doc_filename,
            page=page_number,
            image=str(img_path.resolve()),
            region=bbox_pixels,
            cropped_image_path=str(cropped_out_path.resolve()),
            annotated_image_path=str(annotated_out_path.resolve()),
            region_label=region_label,
        )

    def process_rag_evidence(
        self,
        retrieval_output: RetrievalOutput,
        qa_result: QAAnalysisResult,
    ) -> List[Evidence]:
        """
        处理端到端 RAG 输出，提取证据并生成持久化 Evidence 对象列表
        """
        evidence_list: List[Evidence] = []
        parsed_bboxes = self.parse_bboxes_from_text(qa_result.raw_response or "")

        for idx, item in enumerate(retrieval_output.results):
            page_info = item.page
            bbox_raw = parsed_bboxes[idx][0] if idx < len(parsed_bboxes) else None
            label = parsed_bboxes[idx][1] if idx < len(parsed_bboxes) else f"Page {page_info.page_number} Key Evidence"

            evidence = self.create_evidence(
                document_id=page_info.document_id,
                page_number=page_info.page_number,
                image_path=page_info.image_path,
                bbox_raw=bbox_raw,
                region_label=label,
            )
            evidence_list.append(evidence)

        return evidence_list
