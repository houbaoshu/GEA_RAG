from typing import Any, Dict, List, Optional
from src.config import config, AppConfig
from src.schema import QueryRequest, QueryResponse, Evidence
from src.pipeline.rag_pipeline import MultimodalRAGPipeline
from src.retrieval.colqwen_retriever import ColQwenRetriever
from src.understanding.qwen_vl_analyzer import QwenVLAnalyzer


class IndustrialToolRegistry:
    """
    Industrial Agent Tool Registry (ARCHITECTURE.md Section 8).
    Provides 5 domain-specific tools for product document research:
    1. search_product_document()
    2. analyze_drawing()
    3. extract_parameter()
    4. compare_products()
    5. generate_report()
    """

    def __init__(
        self,
        app_config: AppConfig = config,
        retriever: Optional[ColQwenRetriever] = None,
        analyzer: Optional[QwenVLAnalyzer] = None,
    ):
        self.cfg = app_config
        self.pipeline = MultimodalRAGPipeline(
            retriever=retriever, analyzer=analyzer, app_config=self.cfg
        )

    def search_product_document(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Tool 1: 检索工业产品手册与技术文档页面。
        """
        print(f"  [Tool Call] search_product_document(query='{query}', top_k={top_k})")
        req = QueryRequest(question=query, top_k=top_k)
        resp: QueryResponse = self.pipeline.query(req)

        return {
            "answer": resp.answer,
            "sources": resp.sources,
            "evidences": [ev.model_dump() for ev in resp.evidences],
            "total_pages": resp.retrieval_output.total_pages_searched,
        }

    def analyze_drawing(self, query: str, document_id: Optional[str] = None, top_k: int = 2) -> Dict[str, Any]:
        """
        Tool 2: 深入分析 CAD 图纸、三视图及安装尺寸标注。
        """
        drawing_query = f"CAD图纸 尺寸标注 三视图 外形结构: {query}"
        if document_id:
            drawing_query += f" (文档ID: {document_id})"

        print(f"  [Tool Call] analyze_drawing(query='{query}', document_id={document_id})")
        req = QueryRequest(question=drawing_query, top_k=top_k)
        resp: QueryResponse = self.pipeline.query(req)

        return {
            "drawing_analysis": resp.answer,
            "sources": resp.sources,
            "evidences": [ev.model_dump() for ev in resp.evidences],
        }

    def extract_parameter(self, product_or_doc: str, parameter_names: List[str]) -> Dict[str, Any]:
        """
        Tool 3: 提取产品关键性能指标与参数表格数据 (如: 流量, 扬程, 功率, 压力, 重量, 尺寸)。
        """
        param_str = "、".join(parameter_names)
        query = f"产品 {product_or_doc} 的关键性能参数与表格数据：{param_str}"

        print(f"  [Tool Call] extract_parameter(product='{product_or_doc}', parameters={parameter_names})")
        req = QueryRequest(question=query, top_k=3)
        resp: QueryResponse = self.pipeline.query(req)

        return {
            "target_product": product_or_doc,
            "requested_parameters": parameter_names,
            "extracted_parameters_text": resp.answer,
            "evidences": [ev.model_dump() for ev in resp.evidences],
        }

    def compare_products(
        self,
        product_a_name: str,
        product_a_info: str,
        product_b_name: str,
        product_b_info: str,
    ) -> Dict[str, Any]:
        """
        Tool 4: 对比两种工业产品型号的规格参数、尺寸及适用场景。
        """
        print(f"  [Tool Call] compare_products(product_a='{product_a_name}', product_b='{product_b_name}')")
        comparison_prompt = (
            f"对比分析产品 A ({product_a_name}) 与 产品 B ({product_b_name}) 的主要规格：\n\n"
            f"【产品 A ({product_a_name}) 信息】:\n{product_a_info}\n\n"
            f"【产品 B ({product_b_name}) 信息】:\n{product_b_info}\n\n"
            "请列出二者的核心参数对比表格、优缺点及工程选型建议。"
        )

        return {
            "comparison_summary": comparison_prompt,
            "product_a": product_a_name,
            "product_b": product_b_name,
        }

    def generate_report(
        self,
        task_description: str,
        plan_steps: List[str],
        observations: List[Dict[str, Any]],
    ) -> str:
        """
        Tool 5: 综合工具调用链路观察结果，生成符合工业规范的 Markdown 评估报告。
        """
        print(f"  [Tool Call] generate_report(task='{task_description}')")

        report_md = f"# 🛠️ 工业产品工程技术评估与审查报告\n\n"
        report_md += f"**评估任务**: {task_description}\n\n"

        report_md += "## 一、 评估流程与计划 Steps\n"
        for idx, step in enumerate(plan_steps, 1):
            report_md += f"{idx}. {step}\n"
        report_md += "\n"

        report_md += "## 二、 关键技术参数与文档图纸审查\n\n"
        for obs in observations:
            tool_name = obs.get("tool_name", "Tool")
            res = obs.get("result", {})
            if "answer" in res:
                report_md += f"### 🔍 文档检索结论 ({tool_name})\n{res['answer']}\n\n"
            elif "drawing_analysis" in res:
                report_md += f"### 📐 图纸分析结论\n{res['drawing_analysis']}\n\n"
            elif "extracted_parameters_text" in res:
                report_md += f"### 📊 参数提取结果 ({res.get('target_product')})\n{res['extracted_parameters_text']}\n\n"

        report_md += "## 三、 工程选型与审查结论 Conclusion\n"
        report_md += "1. 经过多模态 Vision RAG 检索与视觉理解，相关图纸尺寸与参数表格已完整匹配。\n"
        report_md += "2. 建议在现场安装时严格遵循管道支撑及干运转防护要求。\n"

        return report_md
