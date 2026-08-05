#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import config


def run_testclient_verification():
    """使用 FastAPI TestClient 对内存服务接口进行全流程断言验证"""
    print("\n📦 初始化 FastAPI TestClient 并执行生命周期预热...")
    from fastapi.testclient import TestClient
    from src.api.app import app

    with TestClient(app) as client:
        # 1. 验证 Health 接口 (无须 Auth)
        print("\n[1/5] 测试 GET /api/v1/health 接口...")
        resp_health = client.get("/api/v1/health")
        assert resp_health.status_code == 200, f"Health 接口状态码异常: {resp_health.status_code}"
        health_data = resp_health.json()
        print(f"  • 服务状态: {health_data['status']}")
        print(f"  • 设备平台: {health_data['device']} ({health_data['dtype']})")
        print(f"  • 检索模型: {health_data['models']['colqwen']}")
        assert health_data["status"] == "healthy"

        # 2. 验证 Authentication 未授权拒绝 (401 Unauthorized)
        print("\n[2/5] 测试 API Auth 身份认证防护 (无 Key / 错 Key 拒绝)...")
        resp_unauth = client.post("/api/v1/query", json={"question": "测试", "top_k": 2})
        assert resp_unauth.status_code == 401, f"未传 API Key 应返回 401，实际返回: {resp_unauth.status_code}"

        resp_badkey = client.post(
            "/api/v1/query",
            json={"question": "测试", "top_k": 2},
            headers={"X-API-Key": "wrong-secret-key"},
        )
        assert resp_badkey.status_code == 401, f"错 API Key 应返回 401，实际返回: {resp_badkey.status_code}"
        print("  • Auth 防护正常！未授权请求已成功被 HTTP 401 拦截。")

        # 3. 验证授权成功的 Vision RAG Query 接口 (POST /api/v1/query)
        print("\n[3/5] 测试带 X-API-Key 的 POST /api/v1/query 检索问答接口...")
        valid_key = config.gea_api_key
        resp_query = client.post(
            "/api/v1/query",
            json={"question": "离心泵操作与安装注意事项有哪些？", "top_k": 2},
            headers={"X-API-Key": valid_key},
        )
        assert resp_query.status_code == 200, f"Query 接口调用失败: {resp_query.text}"
        query_data = resp_query.json()
        print(f"  • 解答内容摘要:\n{query_data['answer'][:150]}...")
        print(f"  • 参考数据源数量: {len(query_data['sources'])}")
        assert len(query_data["answer"].strip()) > 0
        assert len(query_data["sources"]) > 0

        # 4. 验证授权成功的 Agent Task 接口 (POST /api/v1/agent/task)
        print("\n[4/5] 测试带 Bearer Token 的 POST /api/v1/agent/task 接口...")
        resp_agent = client.post(
            "/api/v1/agent/task",
            json={
                "task_description": "审查 GEA 离心泵的安装尺寸与接口规范，提取关键性能参数，并生成工程评估报告",
                "max_steps": 5,
            },
            headers={"Authorization": f"Bearer {valid_key}"},
        )
        assert resp_agent.status_code == 200, f"Agent 接口调用失败: {resp_agent.text}"
        agent_data = resp_agent.json()
        print(f"  • 任务状态: {agent_data['status']}")
        print(f"  • 工作流规划步骤数: {len(agent_data['plan'])}")
        print(f"  • 报告生成状态: {'成功' if agent_data['report'] else '空'}")
        assert agent_data["status"] == "completed"
        assert len(agent_data["execution_trace"]) == 4

        # 5. 验证 Prometheus 监控指标导出 (GET /metrics)
        print("\n[5/5] 测试 GET /metrics Prometheus 监控指标暴露端点...")
        resp_metrics = client.get("/metrics")
        assert resp_metrics.status_code == 200, f"Metrics 端点状态异常: {resp_metrics.status_code}"
        metrics_text = resp_metrics.text
        assert "gea_rag_http_requests_total" in metrics_text
        assert "gea_rag_http_request_duration_seconds" in metrics_text
        print("  • Prometheus 指标暴露正常，包含 HTTP 请求计数器与耗时直方图。")

    print("\n" + "=" * 70)
    print("✅ Milestone 5 验证成功！工业 RESTful API、Auth 认证防护与 Prometheus 监控已全面完成。")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="GEA_RAG Milestone 5 验证脚本 - Industrial API Deployment, Auth & Monitoring 验证"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("🚀 GEA Multimodal RAG - Milestone 5 工业化部署端到端验证")
    print("=" * 70)

    run_testclient_verification()


if __name__ == "__main__":
    main()
