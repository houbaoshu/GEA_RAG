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
        description="GEA_RAG Milestone 2 验证脚本 - Qwen-VL 视觉语言问答全流程"
    )
    parser.add_argument(
        "--question",
        type=str,
        default="离心泵操作与安装注意事项有哪些？",
        help="用户提问语句 (默认: '离心泵操作与安装注意事项有哪些？')",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="检索分析的前 K 个页面图像 (默认: 3)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["local", "api"],
        default="local",
        help="Qwen-VL 推理后端模式: local (本地模型) 或 api (API 接口)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("🚀 GEA Multimodal RAG - Milestone 2 Qwen-VL 视觉理解层 端到端验证")
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
    print("📌 参考数据源 (Sources):")
    for src in response.sources:
        print(f"  • 文档 ID: {src['document']} | 页码: Page {src['page']} | 匹配得分: {src['score']:.4f}")

    print("\n📋 完整 Pydantic / JSON 结构输出 (API 兼容结构):")
    print(response.model_dump_json(indent=2))

    # 3. 结果验证断言
    assert len(response.answer.strip()) > 0, "回答内容不应为空"
    assert len(response.sources) > 0, "参考数据源列表不应为空"
    assert response.sources[0]["page"] > 0, "页码必须为有效正整数"
    assert response.qa_result.evidence_page is not None, "证据页码不可为空"

    print("\n✅ Milestone 2 端到端验证成功！Qwen-VL 视觉理解层已无缝集成。")


if __name__ == "__main__":
    main()
