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

        if not target_path.exists():
            indexer = DocumentIndexer(self.cfg)
            target_path = indexer.ensure_index_exists()

        print(f"📂 加载 ColQwen 多向量索引文件: {target_path.resolve()}")
        index_data = torch.load(target_path, map_location="cpu", weights_only=False)

        raw_image_names = index_data["image_paths"]
        self.page_embeddings = index_data["embeddings"]
        self.model_name = index_data.get("model_name", self.cfg.model_name)

        # 映射图片路径
        self.image_paths = [self.cfg.pages_dir / Path(name).name for name in raw_image_names]

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
