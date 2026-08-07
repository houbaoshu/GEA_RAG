from fastapi import APIRouter, Depends, Request
from src.config import config
from src.schema import QueryRequest, QueryResponse, AgentTaskRequest, AgentTaskResponse
from src.api.auth import verify_api_key
from src.utils.device import get_device_and_dtype

router = APIRouter(prefix="/api/v1", tags=["GEA RAG API"])


@router.get("/health", summary="系统健康检查与服务状态")
async def health_check():
    """返回当前系统运行状态、硬件加速配置与索引状态"""
    device, dtype = get_device_and_dtype()
    index_exists = config.index_path.exists()
    pdf_count = len(list(config.pdf_dir.glob("*.pdf"))) if config.pdf_dir.exists() else 0
    pages_count = len(list(config.pages_dir.glob("*.png"))) if config.pages_dir.exists() else 0

    return {
        "status": "healthy",
        "service": "GEA Multimodal Product Intelligence Agent API",
        "version": "0.1.0",
        "device": device,
        "dtype": str(dtype),
        "models": {
            "colqwen": config.model_name,
            "qwen_vl": config.qwen_vl_model_name,
            "backend": config.qwen_vl_backend,
        },
        "index": {
            "path": str(config.index_path.resolve()),
            "exists": index_exists,
            "pdf_count": pdf_count,
            "pages_count": pages_count,
        },
    }


@router.post("/index/rebuild", summary="扫描 PDF 并重建索引")
async def rebuild_index(
    request: Request,
    api_key: str = Depends(verify_api_key),
):
    """扫描 pdf/ 目录，自动转换 PDF 为图片并重建 ColQwen 特征索引库"""
    from src.pipeline.indexer import DocumentIndexer

    indexer = DocumentIndexer(config)
    indexer.build_index(force_rebuild=True)

    if hasattr(request.app.state, "retriever"):
        request.app.state.retriever.load_index()

    return {
        "status": "success",
        "message": "索引文件重建已完成",
        "index_path": str(config.index_path.resolve()),
    }


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Vision Retrieval & QA 检索问答接口 (ARCHITECTURE.md Section 9)",
)
async def query_rag(
    request: Request,
    body: QueryRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    接收用户提问，执行多模态 Vision Retrieval (ColQwen) 与 Vision Understanding (Qwen-VL)，
    返回包含答案、参考源、证据和图像的标准化 QueryResponse。
    """
    pipeline = request.app.state.rag_pipeline
    response: QueryResponse = pipeline.query(body)
    return response


@router.post(
    "/agent/task",
    response_model=AgentTaskResponse,
    summary="Industrial Agent 自主任务推导接口 (ARCHITECTURE.md Section 8)",
)
async def run_agent_task(
    request: Request,
    body: AgentTaskRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    接收复杂工业工程任务，启动 Autonomous Agent 运行工作流规划、工具调用 (ReAct)
    与工程评估报告生成。
    """
    agent_runtime = request.app.state.agent_runtime
    response: AgentTaskResponse = agent_runtime.run(body)
    return response
