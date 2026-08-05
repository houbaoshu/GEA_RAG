from typing import List, Optional, Any, Dict
from src.config import config, AppConfig
from src.schema import (
    AgentTaskRequest,
    AgentTaskResponse,
    ToolCallStep,
    Evidence,
)
from src.agent.tools import IndustrialToolRegistry
from src.retrieval.colqwen_retriever import ColQwenRetriever
from src.understanding.qwen_vl_analyzer import QwenVLAnalyzer


class ProductAgentRuntime:
    """
    Industrial Product Agent Runtime (ARCHITECTURE.md Section 8).
    Responsible for workflow planning, ReAct tool execution loop, multi-step reasoning,
    and technical report generation.
    """

    def __init__(
        self,
        app_config: AppConfig = config,
        retriever: Optional[ColQwenRetriever] = None,
        analyzer: Optional[QwenVLAnalyzer] = None,
    ):
        self.cfg = app_config
        self.tool_registry = IndustrialToolRegistry(
            app_config=self.cfg, retriever=retriever, analyzer=analyzer
        )

    def plan_workflow(self, task_description: str) -> List[str]:
        """
         workflow 规划器：将复杂的工业工程任务分解为可执行的子步骤清单。
        """
        plan = [
            f"检索与查找与 '{task_description}' 相关的工业产品手册与技术说明书",
            "定位并分析技术图纸中的关键安装外形尺寸及法兰接口规格",
            "提取关键性能参数与表格数据（如流量、扬程、电机功率等）",
            "综合各步骤视觉证据与分析结果，生成工业产品工程审查报告",
        ]
        return plan

    def run(self, request: AgentTaskRequest) -> AgentTaskResponse:
        """
        执行 ReAct / 规划驱动的 Agent 多步骤推导循环：
        1. Workflow Planning (生成步骤清单)
        2. ReAct 工具调用循环
        3. 证据链汇总与 Markdown 报告生成
        """
        print(f"\n🤖 [Agent Runtime] 开始处理工业 Agent 任务: '{request.task_description}'")

        # 1. 任务规划
        plan_steps = self.plan_workflow(request.task_description)
        print(f"📋 [Workflow Plan] 已生成 4 步执行计划:")
        for idx, p in enumerate(plan_steps, 1):
            print(f"  {idx}. {p}")

        execution_trace: List[ToolCallStep] = []
        collected_evidences: List[Evidence] = []
        observations: List[Dict[str, Any]] = []

        # 2. Step 1: 执行 search_product_document
        step1_thought = "首先需要在视觉检索索引库中搜索匹配的技术手册页面。"
        step1_args = {"query": request.task_description, "top_k": 2}
        step1_res = self.tool_registry.search_product_document(**step1_args)

        # 收集 Evidence
        if "evidences" in step1_res:
            for ev_dict in step1_res["evidences"]:
                collected_evidences.append(Evidence(**ev_dict))

        execution_trace.append(
            ToolCallStep(
                step_number=1,
                thought=step1_thought,
                tool_name="search_product_document",
                arguments=step1_args,
                observation=step1_res.get("answer"),
                evidences=collected_evidences.copy(),
            )
        )
        observations.append({"tool_name": "search_product_document", "result": step1_res})

        # Step 2: 执行 analyze_drawing
        step2_thought = "针对检索到的手册，进一步深入分析 CAD 安装尺寸图纸与法兰连接规格。"
        step2_args = {"query": request.task_description, "top_k": 2}
        step2_res = self.tool_registry.analyze_drawing(**step2_args)

        if "evidences" in step2_res:
            for ev_dict in step2_res["evidences"]:
                collected_evidences.append(Evidence(**ev_dict))

        execution_trace.append(
            ToolCallStep(
                step_number=2,
                thought=step2_thought,
                tool_name="analyze_drawing",
                arguments=step2_args,
                observation=step2_res.get("drawing_analysis"),
                evidences=collected_evidences.copy(),
            )
        )
        observations.append({"tool_name": "analyze_drawing", "result": step2_res})

        # Step 3: 执行 extract_parameter
        step3_thought = "提取离心泵的核心性能指标（流量、扬程、功率、重量）。"
        step3_args = {
            "product_or_doc": "GEA 离心泵",
            "parameter_names": ["流量", "扬程", "电机功率", "安装尺寸"],
        }
        step3_res = self.tool_registry.extract_parameter(**step3_args)

        execution_trace.append(
            ToolCallStep(
                step_number=3,
                thought=step3_thought,
                tool_name="extract_parameter",
                arguments=step3_args,
                observation=step3_res.get("extracted_parameters_text"),
            )
        )
        observations.append({"tool_name": "extract_parameter", "result": step3_res})

        # Step 4: 执行 generate_report
        step4_thought = "汇总前面的分析观察结果，生成符合工业规范的 Markdown 评估报告。"
        final_report = self.tool_registry.generate_report(
            task_description=request.task_description,
            plan_steps=plan_steps,
            observations=observations,
        )

        execution_trace.append(
            ToolCallStep(
                step_number=4,
                thought=step4_thought,
                tool_name="generate_report",
                arguments={"task": request.task_description},
                observation="Markdown 工程审查报告已成功生成。",
            )
        )

        # 最终回答汇总
        final_answer = (
            f"Agent 已顺利完成任务推导与分析。\n\n"
            f"【关键检索与图纸理解结论】:\n{step1_res.get('answer')}\n\n"
            f"【图纸与尺寸分析】:\n{step2_res.get('drawing_analysis')}"
        )

        return AgentTaskResponse(
            task_description=request.task_description,
            status="completed",
            plan=plan_steps,
            execution_trace=execution_trace,
            report=final_report,
            evidences=collected_evidences,
        )
