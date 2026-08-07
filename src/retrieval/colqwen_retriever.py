from pathlib import Path
import torch
from src.config import config, AppConfig
from src.schema import DocumentPage, Evidence, SearchResultItem, RetrievalOutput
from src.utils.device import get_device_and_dtype
from src.utils.model_loader import load_colqwen_model
from src.pipeline.indexer import DocumentIndexer
from src.pipeline.pdf_processor import parse_page_metadata


class ColQwenRetriever:
    """Vision Retrieval Engine based on ColQwen Multi-vector Late Interaction (MaxSim)"""

    def __init__(self, app_config: AppConfig = config, auto_load: bool = True):
        self.cfg = app_config
        self.device, self.dtype = get_device_and_dtype()
        self.processor = None
        self.model = None
        self.page_embeddings: list[torch.Tensor] = []
        self.image_paths: list[Path] = []
        self.model_name = self.cfg.model_name

        if auto_load:
            self.load_index()

    def load_index(self, index_path: Path | None = None) -> None:
        """加载多向量索引文件"""
        target_path = index_path or self.cfg.index_path
        indexer = DocumentIndexer(self.cfg)

        # 如果索引不存在，尝试创建
        if not target_path.exists():
            target_path = indexer.ensure_index_exists()

        if target_path.exists():
            print(f"📂 加载 ColQwen 多向量索引文件: {target_path.resolve()}")
            index_data = torch.load(target_path, map_location="cpu", weights_only=False)

            raw_image_names = index_data.get("image_paths", [])
            self.page_embeddings = index_data.get("embeddings", [])
            self.model_name = index_data.get("model_name", self.cfg.model_name)

            # 映射图片路径
            self.image_paths = [self.cfg.pages_dir / Path(name).name for name in raw_image_names]

            # 自动检测：如果索引为空，但 pdf 目录中有 PDF 文件，则自动转换 PDF 并重建索引
            pdf_files = list(self.cfg.pdf_dir.glob("*.pdf")) if self.cfg.pdf_dir.exists() else []
            if len(self.page_embeddings) == 0 and len(pdf_files) > 0:
                print(f"⚠️ 检测到索引为空但存在 {len(pdf_files)} 个 PDF 文件，正在启动自动转换与建库...")
                target_path = indexer.build_index(force_rebuild=True)
                index_data = torch.load(target_path, map_location="cpu", weights_only=False)
                raw_image_names = index_data.get("image_paths", [])
                self.page_embeddings = index_data.get("embeddings", [])
                self.image_paths = [self.cfg.pages_dir / Path(name).name for name in raw_image_names]

        if len(self.page_embeddings) == 0:
            print("⚠️ 注意: 当前索引库为空（未检索到 PDF 文档或页面图片）。请将 PDF 放入 data/pdf 目录并构建索引。")
        else:
            print(f"✅ 索引库加载完成，共 {len(self.image_paths)} 页，模型: {self.model_name}")

        # 延迟/初始化加载 ColQwen 模型与 Processor
        if self.model is None or self.processor is None:
            self.processor, self.model = load_colqwen_model(
                self.model_name, self.device, self.dtype
            )

    def search(self, query: str, top_k: int = 5) -> RetrievalOutput:
        """
        进行 Late Interaction (MaxSim) 多向量检索

        :param query: 文本查询语句
        :param top_k: 返回最匹配的前 K 个页面结果
        :return: 包含标注 Schema 的 RetrievalOutput
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("❌ Retriever 模型尚未初始化，请先调用 load_index()")

        print(f"\n🔍 检索查询: '{query}'")

        if len(self.page_embeddings) == 0:
            print("⚠️ 索引库当前为空，直接返回空结果。")
            return RetrievalOutput(
                query=query,
                top_k=top_k,
                results=[],
                total_pages_searched=0,
            )

        # 1. 提取 Query 多向量特征
        with torch.no_grad():
            batch_query = self.processor.process_queries([query]).to(self.device)
            query_embeddings = self.model(**batch_query)

        query_emb = query_embeddings[0].to(self.device)  # (num_query_tokens, hidden_dim)

        # 2. 计算 Late Interaction (MaxSim) 得分
        scores: list[tuple[Path, float]] = []
        with torch.no_grad():
            for img_path, page_emb in zip(self.image_paths, self.page_embeddings):
                doc_emb = page_emb.to(device=self.device, dtype=query_emb.dtype)
                sim_matrix = torch.matmul(query_emb, doc_emb.T)
                score = sim_matrix.max(dim=-1).values.sum().item()
                scores.append((img_path, score))

        # 3. 降序排列取 Top K
        scores.sort(key=lambda x: x[1], reverse=True)
        top_k_count = min(top_k, len(scores))
        top_scores = scores[:top_k_count]

        # 4. 封装成标准 Pydantic 输出结构 (预留 Milestone 2 Qwen-VL 接口)
        result_items: list[SearchResultItem] = []
        for rank, (img_path, score) in enumerate(top_scores, start=1):
            doc_id, page_num = parse_page_metadata(img_path)

            document_page = DocumentPage(
                document_id=doc_id,
                page_number=page_num,
                image_path=str(img_path),
                metadata={"source_file": img_path.name},
            )

            evidence = Evidence(
                document=f"{doc_id}.pdf" if not doc_id.endswith(".pdf") else doc_id,
                page=page_num,
                image=str(img_path),
                region=None,  # 为 Milestone 3 图像局部 BoundingBox 预留
            )

            result_items.append(
                SearchResultItem(
                    rank=rank,
                    score=round(score, 4),
                    page=document_page,
                    evidence=evidence,
                )
            )

        return RetrievalOutput(
            query=query,
            top_k=top_k,
            results=result_items,
            total_pages_searched=len(self.page_embeddings),
        )
