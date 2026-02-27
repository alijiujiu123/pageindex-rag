FROM python:3.13-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制 pyproject.toml 和 uv.lock
COPY pyproject.toml uv.lock ./

# 安装 uv 并安装依赖
RUN pip install --no-cache-dir uv && \
    uv sync --frozen --no-dev

# 复制应用代码
COPY pageindex/ ./pageindex/
COPY pageindex_rag/ ./pageindex_rag/
COPY scripts/ ./scripts/

# 创建必要的目录
RUN mkdir -p chroma_data results financebench_pdfs

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["uv", "run", "uvicorn", "pageindex_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
