# GEA Multimodal Product Intelligence Agent - Production Dockerfile
FROM python:3.11-slim

# 设置工作目录与环境变量
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GEA_HOST=0.0.0.0 \
    GEA_PORT=8000

# 安装系统依赖 (poppler-utils 用于 PDF 转图, libgl1 用于 OpenCV/PIL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖说明文件并安装 Python 包
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# 复制项目代码与数据
COPY src/ ./src/
COPY docs/ ./docs/
COPY data/ ./data/

# 暴露 API 服务端口
EXPOSE 8000

# 容器健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

# 启动 Uvicorn ASGI 服务
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
