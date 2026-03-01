"""
marker Modal 部署脚本 (Issue #42)

功能：将 marker PDF→MD 转换服务部署到 Modal，使用 L40S GPU 加速。

部署步骤：
    pip install modal
    modal setup                        # 首次认证
    modal run scripts/marker_modal_deployment.py::download_models  # 下载模型（只需一次）
    modal deploy scripts/marker_modal_deployment.py                # 部署服务

调用示例：
    curl -X POST https://<your-url>.modal.run/convert \
      -F "file=@/path/to/doc.pdf" \
      -F "output_format=markdown"
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

        from marker.models import load_all_models

        print("加载 marker 模型...")
        self.models = load_all_models()
        print("✅ 模型加载完成")

    @modal.web_endpoint(method="POST", docs=True)
    async def convert(self, request: dict):
        """
        将 PDF 转换为 Markdown。

        接受 JSON body：
            {
                "pdf_base64": "<base64 编码的 PDF>",
                "output_format": "markdown"   // 可选，默认 markdown
            }

        返回：
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

        from marker.convert import convert_single_pdf
        from marker.output import markdown_exists, save_markdown

        # 解析请求
        pdf_b64 = request.get("pdf_base64")
        if not pdf_b64:
            return {"success": False, "error": "缺少 pdf_base64 字段"}

        output_format = request.get("output_format", "markdown")

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
            # 执行 PDF→MD 转换
            full_text, images, out_meta = convert_single_pdf(
                tmp_path,
                self.models,
                max_pages=None,
                langs=None,
                batch_multiplier=2,
            )
        finally:
            os.unlink(tmp_path)

        elapsed = round(time.time() - start, 2)
        page_count = out_meta.get("pages", 0)

        return {
            "success": True,
            "markdown": full_text,
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
