from typing import Optional, Any
from src.config import config, AppConfig
from src.schema import QueryRequest, QueryResponse, RetrievalOutput, QAAnalysisResult
from src.understanding.qwen_vl_analyzer import QwenVLAnalyzer


class MultimodalRAGPipeline:
    """
    End-to-End Multimodal Industrial RAG Pipeline.
    Combines Vision Retrieval (ColQwen) + Vision Understanding (Qwen-VL).
    """

    def __init__(
        self,
        retriever: Optional[Any] = None,
        analyzer: Optional[QwenVLAnalyzer] = None,
        app_config: AppConfig = config,
    ):
        from src.retrieval.colqwen_retriever import ColQwenRetriever

        self.cfg = app_config
        self.retriever = retriever or ColQwenRetriever(app_config=self.cfg, auto_load=True)
        self.analyzer = analyzer or QwenVLAnalyzer(app_config=self.cfg)

    def query(self, request: QueryRequest) -> QueryResponse:
        """
        全流程端到端问答检索执行：
        1. 视觉检索层：输入问题文本，ColQwen 检索匹配度最高的 Top-K 页面图片。
        2. 视觉理解层：将 Top-K 图片与问题提交给 Qwen-VL，生成多模态视觉解答。
        3. 打包输出符合 ARCHITECTURE.md 规范的 QueryResponse 对象。
        """
        # 1. Vision Retrieval
        retrieval_output: RetrievalOutput = self.retriever.search(
            query=request.question, top_k=request.top_k
        )

        # 2. Vision Understanding
        qa_result: QAAnalysisResult = self.analyzer.analyze(
            query=request.question, retrieval_output=retrieval_output
        )

        # 3. 构造 Sources 引用列表及对应 Evidence 附件信息
        sources = []
        for idx, item in enumerate(retrieval_output.results):
            evidence_item = (
                qa_result.evidences[idx] if idx < len(qa_result.evidences) else None
            )
            sources.append(
                {
                    "page": item.page.page_number,
                    "document": item.page.document_id,
                    "image_path": item.page.image_path,
                    "score": item.score,
                    "cropped_image_path": (
                        evidence_item.cropped_image_path if evidence_item else None
                    ),
                    "annotated_image_path": (
                        evidence_item.annotated_image_path if evidence_item else None
                    ),
                    "region": (
                        evidence_item.region if evidence_item else None
                    ),
                }
            )

        return QueryResponse(
            answer=qa_result.answer,
            sources=sources,
            evidences=qa_result.evidences,
            retrieval_output=retrieval_output,
            qa_result=qa_result,
        )
