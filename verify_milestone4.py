#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.agent import ProductAgentRuntime
from src.schema import AgentTaskRequest, AgentTaskResponse
from src.config import config


def main():
    parser = argparse.ArgumentParser(
        description="GEA_RAG Milestone 4 验证脚本 - Industrial Product Agent System 端到端验证"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="审查 GEA 离心泵的安装尺寸与接口规范，提取关键性能参数，并生成工程评估报告",
        help="用户工程任务描述",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["local", "api"],
        default="local",
        help="Qwen-VL 推理后端模式",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("🚀 GEA Multimodal RAG - Milestone 4 Industrial Agent System 端到端验证")
    print("=" * 70)

    config.qwen_vl_backend = args.backend

    # 1. 初始化 Product Agent Runtime
    agent_runtime = ProductAgentRuntime()

    # 2. 发起 Agent 任务推导
    request = AgentTaskRequest(task_description=args.task)
    response: AgentTaskResponse = agent_runtime.run(request)

    print("\n" + "=" * 70)
    print(f"📌 任务描述: {response.task_description}")
    print(f"🚦 执行状态: {response.status}")
    print("=" * 70)

    print("\n📋 工作流规划 (Workflow Plan):")
    for idx, p in enumerate(response.plan, 1):
        print(f"  {idx}. {p}")

    print("\n🔄 ReAct 执行追踪 (Execution Trace):")
    for step in response.execution_trace:
        print(f"\n  [Step {step.step_number}] 工具: {step.tool_name}")
        print(f"  • 思考 (Thought): {step.thought}")
        print(f"  • 入参 (Arguments): {step.arguments}")
        print(f"  • 状态: {step.status}")

    print("\n" + "=" * 70)
    print("📄 生成的工业工程评估报告 (Generated Markdown Report):")
    print("=" * 70)
    print(response.report)

    # 3. 结果与 Schema 断言验证
    assert response.status == "completed", "Agent 任务状态必须为 completed"
    assert len(response.plan) > 0, "工作流计划不可为空"
    assert len(response.execution_trace) == 4, "工具调用追踪步骤数应为 4"
    assert response.report is not None and len(response.report) > 0, "工程报告不可为空"
    assert len(response.evidences) > 0, "生成的视觉证据链不可为空"

    print("\n✅ Milestone 4 端到端验证成功！Industrial Agent 系统的 Tool Calling、Workflow Planning 与 Report Synthesis 均全面通过。")


if __name__ == "__main__":
    main()
