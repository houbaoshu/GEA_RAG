from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import config
from src.retrieval.colqwen_retriever import ColQwenRetriever
from src.understanding.qwen_vl_analyzer import QwenVLAnalyzer
from src.pipeline.rag_pipeline import MultimodalRAGPipeline
from src.agent.runtime import ProductAgentRuntime
from src.api.routes import router as api_router
from src.api.monitoring import PrometheusMiddleware, get_metrics_response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 应用生命周期预热：在服务启动时预加载并初始化
    ColQwen 检索器、Qwen-VL 理解器与 Agent Runtime 单例，避免请求时的加载开销。
    """
    print("\n🚀 [FastAPI Lifespan] 正在预热初始化 GEA_RAG 核心服务单例...")
    config.ensure_directories()

    # 1. 预加载 ColQwen 检索器与索引
    retriever = ColQwenRetriever(app_config=config, auto_load=True)

    # 2. 预加载 Qwen-VL 分析器
    analyzer = QwenVLAnalyzer(app_config=config)

    # 3. 构造单例流水线与 Agent Runtime
    rag_pipeline = MultimodalRAGPipeline(retriever=retriever, analyzer=analyzer, app_config=config)
    agent_runtime = ProductAgentRuntime(app_config=config, retriever=retriever, analyzer=analyzer)

    # 4. 绑定至 app.state
    app.state.retriever = retriever
    app.state.analyzer = analyzer
    app.state.rag_pipeline = rag_pipeline
    app.state.agent_runtime = agent_runtime

    print("✅ [FastAPI Lifespan] 单例加载完成，API 服务就绪！\n")
    yield
    print("👋 [FastAPI Lifespan] 服务关闭。")


def create_app() -> FastAPI:
    """FastAPI 应用工厂函数 (Milestone 5 API Service)"""
    app = FastAPI(
        title="GEA Multimodal Product Intelligence Agent API",
        description="基于 ColQwen 视觉检索与 Qwen-VL 视觉理解的工业产品智能 Agent REST API",
        version="0.1.0",
        lifespan=lifespan,
    )

    # 1. 配置 CORS 跨域
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. 注册 Prometheus 监控中间件
    app.add_middleware(PrometheusMiddleware)

    # 3. 挂载 Prometheus 监控暴露端点
    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        return get_metrics_response()

    # 4. 挂载 RESTful API 路由
    app.include_router(api_router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.app:app", host=config.api_host, port=config.api_port, reload=False)
