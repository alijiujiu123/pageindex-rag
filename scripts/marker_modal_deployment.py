"""
marker Modal 部署脚本 (Issue #42)

功能：将 marker PDF→JSON/MD 转换服务部署到 Modal，使用 L40S GPU 加速。

部署步骤：
    pip install modal
    modal setup                        # 首次认证
    modal run scripts/marker_modal_deployment.py::download_models  # 下载模型（只需一次）
    modal deploy scripts/marker_modal_deployment.py                # 部署服务

调用示例（JSON 模式，推荐）：
    curl -X POST https://<your-url>.modal.run/convert \
      -H "Content-Type: application/json" \
      -d '{"pdf_base64": "<base64>", "output_format": "json"}'

调用示例（Markdown 模式）：
    curl -X POST https://<your-url>.modal.run/convert \
      -H "Content-Type: application/json" \
      -d '{"pdf_base64": "<base64>", "output_format": "markdown"}'
"""

import modal

# ── 常量 ──────────────────────────────────────────────────────────────────────
APP_NAME = "pageindex-rag-marker"
VOLUME_NAME = "pageindex-rag-marker-models"
MODEL_DIR = "/models"
GPU_TYPE = "L40S"

# ── Modal 资源 ─────────────────────────────────────────────────────────────────
app = modal.App(APP_NAME)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "marker-pdf>=0.2.0",
        "fastapi>=0.110.0",
        "python-multipart>=0.0.9",
    )
)


# ── 模型下载（一次性任务）──────────────────────────────────────────────────────
@app.function(
    image=image,
    volumes={MODEL_DIR: volume},
    gpu=GPU_TYPE,
    timeout=3600,
)
def download_models():
    """下载 marker 所需模型到持久化 Volume（只需运行一次）。"""
    import os

    # marker 首次推理时会自动下载模型；我们通过运行一次空转来触发下载
    os.environ["MARKER_MODEL_DIR"] = MODEL_DIR

    from marker.models import load_all_models

    print("开始下载 marker 模型...")
    models = load_all_models()
    volume.commit()
    print(f"✅ 模型下载完成，保存到 {MODEL_DIR}")
    return {"status": "ok", "model_dir": MODEL_DIR}


# ── FastAPI 服务 ───────────────────────────────────────────────────────────────
@app.cls(
    image=image,
    volumes={MODEL_DIR: volume},
    gpu=GPU_TYPE,
    timeout=600,
    container_idle_timeout=300,  # 空闲 5 分钟后缩容，节省费用
)
class MarkerService:
    @modal.enter()
    def load_models(self):
        """容器启动时加载模型（热启动复用）。"""
        import os

        os.environ["MARKER_MODEL_DIR"] = MODEL_DIR
        print("✅ 模型将在首次请求时按需加载")

    @modal.web_endpoint(method="POST", docs=True)
    async def convert(self, request: dict):
        """
        将 PDF 转换为 Markdown 或 JSON。

        接受 JSON body：
            {
                "pdf_base64": "<base64 编码的 PDF>",
                "output_format": "json" | "markdown"   // 可选，默认 json
            }

        返回（json 模式）：
            {
                "success": true,
                "marker_json": { /* JSONOutput dict */ },
                "page_count": N,
                "elapsed_seconds": X.X
            }

        返回（markdown 模式）：
            {
                "success": true,
                "markdown": "...",
                "page_count": N,
                "elapsed_seconds": X.X
            }
        """
        import base64
        import os
        import tempfile
        import time

        # 解析请求
        pdf_b64 = request.get("pdf_base64")
        if not pdf_b64:
            return {"success": False, "error": "缺少 pdf_base64 字段"}

        output_format = request.get("output_format", "json")
        if output_format not in ("json", "markdown"):
            return {"success": False, "error": f"不支持的 output_format: {output_format}，仅支持 json/markdown"}

        # 解码 PDF 到临时文件
        try:
            pdf_bytes = base64.b64decode(pdf_b64)
        except Exception as e:
            return {"success": False, "error": f"base64 解码失败: {e}"}

        start = time.time()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict

            converter = PdfConverter(
                artifact_dict=create_model_dict(),
                renderer=output_format,
            )
            rendered = converter(tmp_path)
        finally:
            os.unlink(tmp_path)

        elapsed = round(time.time() - start, 2)

        if output_format == "json":
            # rendered 是 JSONOutput 对象
            import dataclasses
            marker_json = rendered.model_dump() if hasattr(rendered, "model_dump") else dataclasses.asdict(rendered)
            page_count = marker_json.get("metadata", {}).get("pages", 0)
            return {
                "success": True,
                "marker_json": marker_json,
                "page_count": page_count,
                "elapsed_seconds": elapsed,
            }
        else:
            # rendered 是 MarkdownOutput 对象
            markdown = rendered.markdown if hasattr(rendered, "markdown") else str(rendered)
            page_count = rendered.metadata.get("pages", 0) if hasattr(rendered, "metadata") else 0
            return {
                "success": True,
                "markdown": markdown,
                "page_count": page_count,
                "elapsed_seconds": elapsed,
            }


# ── 本地测试入口 ───────────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    """本地快速测试（需要已部署的服务）。"""
    print(f"App: {APP_NAME}")
    print(f"Volume: {VOLUME_NAME}")
    print("使用以下命令部署：")
    print("  modal run scripts/marker_modal_deployment.py::download_models")
    print("  modal deploy scripts/marker_modal_deployment.py")
