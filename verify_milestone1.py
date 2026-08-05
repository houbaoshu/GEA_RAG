#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# 保证能正确 import src 目录下的模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.retrieval import ColQwenRetriever
from src.pipeline import DocumentIndexer
from src.schema import RetrievalOutput


def main():
    parser = argparse.ArgumentParser(
        description="GEA_RAG Milestone 1 验证脚本 - ColQwen3.5 视觉检索"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="离心泵操作指南",
        help="搜索查询语句 (默认: '离心泵操作指南')",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="返回匹配度最高的 Top-K 页面路径 (默认: 5)",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="强制重新提取特征生成 index.pt 索引文件",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("🚀 GEA Multimodal RAG - Milestone 1 Vision Retrieval Core 验证")
    print("=" * 70)

    # 1. 索引检测/重构
    indexer = DocumentIndexer()
    if args.rebuild_index:
        print("🔄 正在强制重新构建多向量索引...")
        indexer.build_index(force_rebuild=True)
    else:
        indexer.ensure_index_exists()

    # 2. 初始化 Vision Retriever
    retriever = ColQwenRetriever(auto_load=True)

    # 3. 执行检索
    output: RetrievalOutput = retriever.search(query=args.query, top_k=args.top_k)

    print("\n" + "=" * 70)
    print(f"🏆 Top-{output.top_k} 检索结果汇总 (查询: '{output.query}')")
    print("=" * 70)

    for result in output.results:
        print(
            f"Rank {result.rank:2d} | 得分: {result.score:7.4f} | "
            f"文档: {result.page.document_id:10s} | 页码: Page {result.page.page_number:3d} | "
            f"图片路径: {result.page.image_path}"
        )

    print("-" * 70)
    print("\n📋 Pydantic 导出的标准 JSON 输出 (预留 Milestone 2 Qwen-VL 接口):")
    print(output.model_dump_json(indent=2))

    # 4. Milestone 2 / 3 兼容性断言验证
    assert len(output.results) > 0, "检索结果不应为空"
    for item in output.results:
        assert Path(item.page.image_path).exists(), f"图片路径不存在: {item.page.image_path}"
        assert item.evidence.document is not None, "Evidence 对象 document 不可为空"
        assert item.evidence.page > 0, "Evidence 对象 page 必须为正整数"

    print("\n✅ Milestone 1 验证成功！所有输出均符合 ARCHITECTURE.md 规定的标准接口规格。")


if __name__ == "__main__":
    main()
