from typing import Any, Optional, List
from pydantic import BaseModel, Field


class DocumentPage(BaseModel):
    """文档页面核心对象 (ARCHITECTURE.md Section 4.1)"""
    document_id: str = Field(description="文档标识符或源 PDF 文件名")
    page_number: int = Field(description="1-based 页面序号")
    image_path: str = Field(description="渲染导出的 PNG 图片路径")
    metadata: dict[str, Any] = Field(default_factory=dict, description="扩展元数据（如产品型号、文档来源等）")


class Evidence(BaseModel):
    """证据对象 (ARCHITECTURE.md Section 7)"""
    document: str = Field(description="源 PDF 文件名称")
    page: int = Field(description="页码引用")
    image: str = Field(description="页面原始图片路径")
    region: Optional[List[int] | str] = Field(
        default=None,
        description="图纸/表格区域像素坐标框 [x1, y1, x2, y2] 或区域描述"
    )
    cropped_image_path: Optional[str] = Field(
        default=None,
        description="自动裁剪导出的局部证据图片路径"
    )
    annotated_image_path: Optional[str] = Field(
        default=None,
        description="带有高亮 BoundingBox 标注的整页图片路径"
    )
    region_label: Optional[str] = Field(
        default=None,
        description="区域类型描述 (如: 尺寸图纸、参数表格、警告说明等)"
    )


class SearchResultItem(BaseModel):
    """检索命中单项"""
    rank: int = Field(description="1-based 匹配排名")
    score: float = Field(description="Late Interaction (MaxSim) 相似度得分")
    page: DocumentPage = Field(description="匹配的页面详情")
    evidence: Evidence = Field(description="为 Qwen-VL 视觉理解准备的证据接口结构")


class RetrievalOutput(BaseModel):
    """Vision Retrieval 标准输出结构 (Milestone 1 Core Output)"""
    query: str = Field(description="用户检索查询语句")
    top_k: int = Field(description="请求的 Top-K 数量")
    results: List[SearchResultItem] = Field(default_factory=list, description="结果列表")
    total_pages_searched: int = Field(description="检索库包含的总页面数")


class QAAnalysisResult(BaseModel):
    """Qwen-VL 视觉分析层输出结构 (Milestone 2/3 Output)"""
    answer: str = Field(description="Qwen-VL 结合图像与问题分析生成的回答")
    evidence_page: Optional[int] = Field(default=None, description="最核心证据页码")
    evidence_region: Optional[str] = Field(default=None, description="定位到的关键区域描述或图表说明")
    evidences: List[Evidence] = Field(default_factory=list, description="生成的完整证据对象列表 (Milestone 3)")
    raw_response: Optional[str] = Field(default=None, description="模型原始输出文本")


class QueryRequest(BaseModel):
    """API Query 请求体 (ARCHITECTURE.md Section 9)"""
    question: str = Field(description="用户提问语句")
    top_k: int = Field(default=3, description="检索匹配的页面图片数")


class QueryResponse(BaseModel):
    """API Query 响应体 (ARCHITECTURE.md Section 9)"""
    answer: str = Field(description="多模态 RAG 最终生成的解答")
    sources: List[dict[str, Any]] = Field(default_factory=list, description="参考源引用列表")
    evidences: List[Evidence] = Field(default_factory=list, description="详细视觉证据对象列表 (Milestone 3)")
    retrieval_output: RetrievalOutput = Field(description="Milestone 1 检索输出详情")
    qa_result: QAAnalysisResult = Field(description="Milestone 2 视觉理解详细分析结果")

