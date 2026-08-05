import time
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# 定义 Prometheus 监控指标
REQUEST_COUNT = Counter(
    "gea_rag_http_requests_total",
    "HTTP Request Count",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "gea_rag_http_request_duration_seconds",
    "HTTP Request Duration in Seconds",
    ["method", "endpoint"],
)

QUERY_LATENCY = Histogram(
    "gea_rag_retrieval_query_duration_seconds",
    "ColQwen Retrieval Query Latency in Seconds",
)

AGENT_STEP_COUNT = Counter(
    "gea_rag_agent_tool_calls_total",
    "Agent System Tool Call Count",
    ["tool_name"],
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """请求耗时与计数 Prometheus 监控中间件 (Milestone 5 Monitoring)"""

    async def dispatch(self, request: Request, call_next) -> Response:
        endpoint = request.url.path
        method = request.method

        start_time = time.time()
        try:
            response = await call_next(request)
            status_code = str(response.status_code)
        except Exception as exc:
            status_code = "500"
            raise exc from None
        finally:
            duration = time.time() - start_time
            REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
            REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)

        return response


def get_metrics_response() -> Response:
    """导出 Prometheus 文本格式监控数据 (/metrics)"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
