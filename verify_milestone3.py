#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline.rag_pipeline import MultimodalRAGPipeline
from src.schema import QueryRequest, QueryResponse
from src.config import config


def main():
    parser = argparse.ArgumentParser(
        description="GEA_RAG Milestone 3 验证脚本 - 视觉证据链 (Evidence System) 验证"
    )
    parser.add_argument(
        "--question",
        type=str,
        default="离心泵安装尺寸与法兰接口规格说明",
        help="用户提问语句",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="检索分析的前 K 个页面图像",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["local", "api"],
        default="local",
        help="Qwen-VL 推理后端模式",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("🚀 GEA Multimodal RAG - Milestone 3 Evidence System 视觉证据链端到端验证")
    print("=" * 70)

    config.qwen_vl_backend = args.backend

    # 1. 初始化端到端 RAG Pipeline
    pipeline = MultimodalRAGPipeline()

    # 2. 发起检索与问答
    request = QueryRequest(question=args.question, top_k=args.top_k)
    response: QueryResponse = pipeline.query(request)

    print("\n" + "=" * 70)
    print(f"💬 问题: {args.question}")
    print("=" * 70)
    print(f"💡 Qwen-VL 视觉解答:\n{response.answer}\n")
    print("-" * 70)

    print("🔍 视觉证据链 (Evidence Objects):")
    for idx, ev in enumerate(response.evidences, start=1):
        print(f"\n  [证据 {idx}]")
        print(f"  • 源文档: {ev.document}")
        print(f"  • 对应页码: Page {ev.page}")
        print(f"  • 原始整页图片: {ev.image}")
        print(f"  • BoundingBox 像素坐标: {ev.region}")
        print(f"  • 自动裁剪局部图片: {ev.cropped_image_path}")
        print(f"  • 红框标注整页图片: {ev.annotated_image_path}")

    print("\n📋 完整 Pydantic / JSON 结构输出 (带 Evidence 字段):")
    print(response.model_dump_json(indent=2))

    # 3. 结果与物理文件断言验证
    assert len(response.answer.strip()) > 0, "回答内容不应为空"
    assert len(response.evidences) > 0, "生成 Evidence 对象列表不应为空"

    for ev in response.evidences:
        assert ev.document is not None and len(ev.document) > 0, "文档名称必须有效"
        assert ev.page > 0, "页码必须为有效正整数"
        assert ev.cropped_image_path is not None, "局部裁剪图片路径不可为空"
        assert Path(ev.cropped_image_path).exists(), f"裁剪图片文件不存在: {ev.cropped_image_path}"
        assert ev.annotated_image_path is not None, "标注图片路径不可为空"
        assert Path(ev.annotated_image_path).exists(), f"标注图片文件不存在: {ev.annotated_image_path}"
        assert ev.region is not None and len(ev.region) == 4, "BoundingBox 坐标必须为 4 维数组"

    print("\n✅ Milestone 3 端到端验证成功！视觉证据链、自动裁剪与 BoundingBox 标注已全面通过。")


if __name__ == "__main__":
    main()
