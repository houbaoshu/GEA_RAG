# GEA Multimodal Product Intelligence Agent (GEA-RAG)

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-FF5500?style=flat-square&logo=gradio&logoColor=white)](https://gradio.app/)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)

GEA Multimodal Product Intelligence Agent 是一个基于**视觉文档检索 (Vision Space Retrieval)** 与 **视觉语言模型 (VLM)** 的工业产品智能助手。

专为复杂工业 PDF 文档（产品手册、技术规格书、CAD 图纸、三视图、安装说明书、参数表等）设计。系统通过将 PDF 页面直接转化为视觉特征向量，避免了传统 PDF RAG 在 OCR 文本提取时导致的**表格结构丢失、图纸细节失效、矢量标注错位**等瓶颈问题。

---

## 🌟 核心特性

- **视觉端到端检索 (Vision Retrieval)**：集成 **ColQwen3.5** 多向量 (Multi-vector) 视觉特征提取与 Late Interaction 晚期交互比对，直接对 PDF 页面图像进行精准语义及空间检索。
- **视觉语言深度理解 (Vision QA & Analysis)**：采用 **Qwen2.5-VL-3B-Instruct**（支持本地推理与 `qwen-vl-max` 云端 API 双后端自动降级与切换），实现对复杂 CAD 图纸、三视图尺寸、部件标注与结构表格的高精度读取与问答。
- **可追溯视觉证据链 (Evidence System)**：生成高亮红框标注整页图（Annotated Page）与局部区域放大裁剪图（Cropped Region），支持精准的地理坐标（Bounding Box）回溯与可视化证明。
- **工业 ReAct Agent 自主规划**：内建自主工程 Agent 运行时（Product Agent Runtime），支持复杂任务分解、工具调用（`search_product_document`, `analyze_drawing`, `extract_parameter`, `compare_products`, `generate_report`）及自动工程审查报告生成。
- **双端交互界面与工业部署**：
  - **Gradio Web UI**：直观的多页签可视化前端，支持参数调节、样例快捷加载与双图层证据链展示。
  - **FastAPI 工业级 RESTful API**：支持 Bearer API Key 身份鉴权、Prometheus 运维指标监控及 Docker / Docker-Compose 一键部署。
- **自动化 Benchmark 评测套件**：内嵌基于真实工程文档的评测集与评估脚本，自动量化 Recall@1/3、尺寸准确率、物理单位保留率及 BBox IoU。

---

## 🏗️ 架构理念与对比

### 传统 PDF RAG vs 本系统

```
【传统 PDF RAG】
PDF ➔ OCR ➔ 文本解析 ➔ Chunk 切片 ➔ Text Embedding ➔ 向量检索 ➔ LLM
❌ 缺点：表格格式混乱、CAD 图纸与三视图细节丢失、尺寸标注错位

【GEA Vision RAG (本系统)】
PDF ➔ Page Image ➔ ColQwen Multi-vector ➔ Visual Top-K ➔ Qwen-VL ➔ 结构化解答+证据链
✅ 优势：保留完整视觉排版、直接读取几何图案与尺寸表格、生成准确 Bounding Box 证据
```

---

## 📁 目录结构

```text
GEA_RAG/
├── app_ui.py                   # Gradio Web 可视化 UI 启动入口
├── docker-compose.yml          # Docker Compose 组网部署配置 (API + Prometheus)
├── Dockerfile                  # 生产级 Docker 镜像构建文件
├── pyproject.toml              # 项目依赖与 Python 环境配置
├── prometheus.yml              # Prometheus 监控采集配置
├── src/                        # 核心源代码目录
│   ├── agent/                  # Milestone 4: ReAct Agent 运行时与工具链
│   │   ├── runtime.py          # ProductAgentRuntime
│   │   └── tools.py            # Agent 工具集中管理
│   ├── api/                    # Milestone 5: FastAPI 接口与中间件
│   │   ├── app.py              # FastAPI 应用与生命周期管理
│   │   ├── auth.py             # API Key 鉴权中间件
│   │   ├── monitoring.py       # Prometheus 监控指标配置
│   │   └── routes.py           # RESTful 路由定义 (/health, /query, /agent/task)
│   ├── evidence/               # Milestone 3: 视觉证据链管理
│   │   └── evidence_manager.py # 坐标定位、画框标注与局部放大裁剪
│   ├── pipeline/               # Milestone 1 & 2: RAG 流水线与文档索引
│   │   ├── indexer.py          # 页面图像索引构建器
│   │   ├── pdf_processor.py    # PDF 转图像处理器
│   │   └── rag_pipeline.py     # 视觉 RAG 端到端串联流水线
│   ├── retrieval/              # Milestone 1: ColQwen 视觉检索器
│   │   └── colqwen_retriever.py
│   ├── understanding/          # Milestone 2: Qwen-VL 视觉理解分析器
│   │   └── qwen_vl_analyzer.py # 支持 本地与 API 模式及回退机制
│   ├── ui/                     # Web UI 模块
│   │   └── app_ui.py           # Gradio Blocks 界面设计
│   ├── config.py               # 全局配置中心 (Pydantic / dotenv)
│   ├── schema.py               # 统一数据结构定义 (Pydantic Models)
│   └── utils/                  # 工具类 (设备判断、图像处理、模型加载)
├── scripts/
│   └── generate_benchmark_dataset.py # Benchmark 评估数据集生成脚本
├── tests/
│   ├── eval_benchmark.py       # 自动化 Benchmark 指标评估套件
│   └── benchmark_dataset.json  # 工业测试用例集
├── docs/
│   └── ARCHITECTURE.md         # 详细架构与技术设计文档
└── verify_milestone[1-5].py    # 里程碑 1~5 模块断言验证脚本
```

---

## 🛠️ 环境准备与安装

### 1. 系统基础依赖

依赖 `poppler-utils`（用于 PDF 转图）及 `libgl1`（OpenCV 依赖）：

- **macOS**:
  ```bash
  brew install poppler
  ```
- **Ubuntu/Debian**:
  ```bash
  sudo apt-get update && sudo apt-get install -y poppler-utils libgl1 libglib2.0-0
  ```

### 2. Python 环境搭建

推荐使用 Python 3.11+ 及 `uv` 虚拟环境包管理器：

```bash
# 使用 uv 创建并激活虚拟环境
uv venv .venv
source .venv/bin/activate

# 安装项目依赖
uv pip install -e .
```

*（也可以使用原生 pip：`pip install -e .`）*

### 3. 环境变量配置 (`.env`)

在项目根目录下创建或修改 `.env` 文件：

```env
# API Key 鉴权密钥 (生产环境请修改)
GEA_API_KEY=gea-industrial-secret-key

# Qwen-VL 模型后端选择: "local" (本地 PyTorch 推理) 或 "api" (阿里云 DashScope)
QWEN_VL_BACKEND=api

# DashScope / 通义千问 API 密钥 (API 后端模式下使用)
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

---

## 🚀 快速上手与运行说明

### 1. Gradio Web 可视化界面 (推荐)

启动端到端 Web 交互界面：

```bash
python app_ui.py
```

访问 `http://127.0.0.1:7860` 即可在浏览器中体验：
- 🔍 输入工业问题并调节 Top-K 检索数
- 💡 查看 Qwen-VL 视觉理解解答及元数据引用
- 🔎 浏览红框标注整页图与局部细节放大图

---

### 2. FastAPI 后端服务

启动生产 API 服务：

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

- **Health 检查**: `GET http://localhost:8000/api/v1/health`
- **Swagger 接口文档**: `http://localhost:8000/docs`
- **Prometheus Metrics**: `http://localhost:8000/metrics`

---

### 3. Docker 一键部署

构建并启动 API 服务与 Prometheus 监控套件：

```bash
# 后台启动容器组
docker-compose up -d --build

# 查看运行状态
docker-compose ps

# 检查 Health 状态
curl -f http://localhost:8000/api/v1/health
```

---

### 4. 里程碑断言验证脚本

项目提供 5 个独立的阶段性验证脚本，方便快速测试各个关键模块：

```bash
# Milestone 1: 视觉检索测试 (ColQwen)
python verify_milestone1.py --query "离心泵操作指南" --top-k 3

# Milestone 2: 视觉问答测试 (RAG Pipeline + Qwen-VL)
python verify_milestone2.py --question "离心泵操作与安装注意事项有哪些？"

# Milestone 3: 视觉证据链测试 (Evidence BBox & Crop)
python verify_milestone3.py --question "离心泵安装尺寸与法兰接口规格说明"

# Milestone 4: Autonomous Agent 自主任务规划测试
python verify_milestone4.py --task "审查 GEA 离心泵安装尺寸，生成工程评估报告"

# Milestone 5: API 接口与鉴权集成测试 (FastAPI TestClient)
python verify_milestone5.py
```

---

## 📊 自动化 Benchmark 评测

项目内置针对工业 PDF 场景的 Benchmark 测试框架（支持评估 4 大关键工程指标）：

```bash
python tests/eval_benchmark.py
```

**评估指标说明：**
1. **Retrieval Recall@1 / Recall@3**：ColQwen 视觉检索页面命中率。
2. **Dimension Accuracy**：数值与安装尺寸提取准确率。
3. **Unit Preserved Rate**：物理单位（如 `mm`, `m³/h`, `℃`）保留完整率。
4. **BBox IoU**：证据定位红框坐标与 Ground Truth 的空间交并比均值及 Pass@0.5。

---

## 📡 API 接口规范

所有的标准 RESTful 接口均位于 `/api/v1` 前缀下，使用 `Authorization: Bearer <GEA_API_KEY>` 请求头鉴权。

### 1. RAG 视觉检索问答 `/api/v1/query` [POST]

**Request Body:**
```json
{
  "question": "GEA 离心泵进口法兰 Suction Port 规格是多少？",
  "top_k": 3
}
```

**Response:**
```json
{
  "answer": "根据 GEA 离心泵规格说明书 Page 2，进口法兰规格为 DN50 PN16...",
  "sources": [
    {
      "document": "GEA_Pump_Manual.pdf",
      "page": 2,
      "score": 0.892,
      "annotated_image_path": "/app/data/evidence/ann_page_2.png",
      "cropped_image_path": "/app/data/evidence/crop_page_2.png"
    }
  ],
  "evidence": [
    {
      "document": "GEA_Pump_Manual.pdf",
      "page": 2,
      "bbox": [120, 340, 580, 720]
    }
  ]
}
```

### 2. Autonomous Agent 任务接口 `/api/v1/agent/task` [POST]

**Request Body:**
```json
{
  "task": "审查离心泵安装尺寸并评估接口适配性",
  "max_steps": 5
}
```

---

## 🛣️ 项目里程碑路线图

- [x] **Milestone 1**: 视觉检索核心（PDF 转图、ColQwen 多向量索引与 Late Interaction 检索）
- [x] **Milestone 2**: 视觉问答流水线（Qwen-VL 本地/API 模式融合与结构化回答生成）
- [x] **Milestone 3**: 视觉证据链系统（BBox 定位、全页红框高亮与细节裁剪）
- [x] **Milestone 4**: Industrial Agent 运行时（ReAct 架构、工具链与工程报告生成）
- [x] **Milestone 5**: 工业级部署与监控（FastAPI 路由、Bearer 鉴权、Prometheus 监控与 Docker Compose）
- [x] **Benchmark**: 自动化 4 大工程指标评测套件与 Web UI 前端集成

---

## 📄 开源许可证

本项目基于 [MIT License](LICENSE) 许可发布。
