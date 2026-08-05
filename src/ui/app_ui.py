#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import gradio as gr

# 确保正确引用项目根目录下的 src 模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import config
from src.schema import QueryRequest, QueryResponse
from src.pipeline.rag_pipeline import MultimodalRAGPipeline

# 单例全局 MultimodalRAGPipeline
_pipeline_instance: Optional[MultimodalRAGPipeline] = None


def get_pipeline() -> MultimodalRAGPipeline:
    """获取 MultimodalRAGPipeline 单例实例，避免每次提问重复初始化模型"""
    global _pipeline_instance
    if _pipeline_instance is None:
        import os

        print("\n🚀 正在预热加载 MultimodalRAGPipeline 单例引擎 (API 后端模式)...")
        config.ensure_directories()
        config.qwen_vl_backend = "api"

        # 校验 API Key 状态
        api_key = (
            os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("QWEN_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if api_key:
            masked_key = (
                api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "***"
            )
            print(f"🔑 已检测并激活 Qwen-VL API 密钥: {masked_key}")
        else:
            print(
                "⚠️ 未检测到 DASHSCOPE_API_KEY / QWEN_API_KEY，请检查 .env 或环境变量配置！"
            )

        _pipeline_instance = MultimodalRAGPipeline(app_config=config)
        print("✅ 流水线引擎加载完成！\n")
    return _pipeline_instance


def process_query(
    question: str, top_k: int
) -> Tuple[str, List[Dict[str, Any]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Gradio 交互回调处理函数：
    接收提问与 Top-K 数量，调用 RAG Pipeline 检索与问答，
    返回 (Markdown 答案, Sources JSON, 高亮整页标注图列表, 局部裁剪放大细节图列表)
    """
    if not question or not question.strip():
        return "⚠️ 请输入具体的工业文档提问内容！", [], [], []

    pipeline = get_pipeline()
    request = QueryRequest(question=question.strip(), top_k=int(top_k))
    response: QueryResponse = pipeline.query(request)

    answer_md = response.answer
    sources_json = response.sources

    annotated_gallery: List[Tuple[str, str]] = []
    cropped_gallery: List[Tuple[str, str]] = []

    for src in response.sources:
        doc = src.get("document", "Unknown")
        page = src.get("page", 0)
        score = src.get("score", 0.0)

        # 高亮红框标注整页图
        ann_path = src.get("annotated_image_path") or src.get("image_path")
        if ann_path and Path(ann_path).exists():
            caption = f"文档: {doc} | 页码: Page {page} | 匹配得分: {score:.4f}"
            annotated_gallery.append((str(Path(ann_path).resolve()), caption))

        # 局部放大细节图
        crop_path = src.get("cropped_image_path")
        if crop_path and Path(crop_path).exists():
            caption = f"局部细节: Document {doc} Page {page}"
            cropped_gallery.append((str(Path(crop_path).resolve()), caption))

    return answer_md, sources_json, annotated_gallery, cropped_gallery


def create_ui() -> gr.Blocks:
    """构建 Gradio 可视化 UI 页面 (Milestone 6 / 方案 B Web UI)"""
    theme = gr.themes.Soft()

    with gr.Blocks(
        title="GEA Multimodal Product Intelligence Agent", theme=theme
    ) as demo:
        # 顶部标题与项目简介
        gr.Markdown(
            """
            # 🏭 GEA Multimodal Product Intelligence Agent
            ### 基于 ColQwen3.5 视觉检索与 Qwen2.5-VL 视觉理解的工业产品智能助手
            *支持复杂 PDF 手册、技术规格书、CAD 图纸与尺寸参数表的端到端视觉检索问答*
            """
        )

        with gr.Row():
            # 左侧控制栏
            with gr.Column(scale=1):
                gr.Markdown("### 🔍 提问与控制面板")
                question_input = gr.Textbox(
                    label="工业提问 / 参数查询",
                    placeholder="例如：离心泵基础固定螺栓孔距是多少？",
                    lines=3,
                )
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    label="Top-K 页面数",
                )
                submit_btn = gr.Button("🚀 发送提问", variant="primary")

                gr.Markdown("#### 💡 预设样例问题 (点击加载)")
                preset_examples = [
                    ["离心泵基础固定螺栓孔距是多少？", 3],
                    ["GEA 离心泵进口法兰 Suction Port 规格是多少？", 3],
                    ["离心泵操作与安装注意事项有哪些？", 2],
                    ["GEA 离心泵允许工作的最高介质温度是多少？", 3],
                ]

                # 快速点击预设问题按钮
                for example_text, example_top_k in preset_examples:
                    btn = gr.Button(f"📌 {example_text}", size="sm")
                    btn.click(
                        fn=lambda q=example_text, k=example_top_k: (q, k),
                        outputs=[question_input, top_k_slider],
                    )

            # 右侧主面板 (Tabs)
            with gr.Column(scale=2):
                with gr.Tabs():
                    # Tab 1: 智能解答区
                    with gr.Tab("💡 智能解答 (Smart Answer)"):
                        answer_output = gr.Markdown(
                            label="Qwen-VL 视觉理解解答",
                            value="*请输入问题并点击【发送提问】按钮...*",
                        )
                        sources_output = gr.JSON(
                            label="参考数据源元数据 (Sources Citation)",
                            value=[],
                        )

                    # Tab 2: 视觉证据链区
                    with gr.Tab("🔍 视觉证据链 (Evidence Viewer)"):
                        gr.Markdown("#### 📄 高亮标注整页图 (Annotated Full Pages)")
                        annotated_gallery = gr.Gallery(
                            label="红框标注页面",
                            columns=2,
                            height="auto",
                            object_fit="contain",
                        )
                        gr.Markdown("#### 🔎 局部细节放大图 (Cropped Detail Regions)")
                        cropped_gallery = gr.Gallery(
                            label="局部放大部分",
                            columns=2,
                            height="auto",
                            object_fit="contain",
                        )

        # 触发提交事件
        submit_btn.click(
            fn=process_query,
            inputs=[question_input, top_k_slider],
            outputs=[answer_output, sources_output, annotated_gallery, cropped_gallery],
        )

    return demo


def main():
    print("=" * 70)
    print("🌐 启动 GEA Multimodal Product Intelligence Agent Web UI Server...")
    print("=" * 70)
    demo = create_ui()
    server_name = "127.0.0.1"
    server_port = 7860
    print(f"🔗 本地 Web 服务访问地址: http://{server_name}:{server_port}")
    demo.launch(server_name=server_name, server_port=server_port, share=True)


if __name__ == "__main__":
    main()
