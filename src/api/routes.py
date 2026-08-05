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
        },
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
