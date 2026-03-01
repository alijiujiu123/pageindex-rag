"""
marker Modal 验证脚本 (Issue #42)

功能：
    1. 上传 PDF 到 Modal marker 服务，获取 Markdown
    2. 保存结果到 output/marker_<filename>.md
    3. 打印耗时、页数等统计信息

用法：
    MARKER_MODAL_URL=https://xxx.modal.run \
        python scripts/test_marker_modal.py --pdf /path/to/test.pdf

    # 也可在 .env 中设置 MARKER_MODAL_URL
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path


def load_env():
    """简单读取项目根目录的 .env 文件（不依赖 python-dotenv）。"""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())


def convert_pdf(modal_url: str, pdf_path: Path) -> dict:
    """
    上传 PDF 到 Modal marker 服务并返回响应。

    Args:
        modal_url: Modal 服务 URL（无尾斜杠）
        pdf_path: 本地 PDF 文件路径

    Returns:
        服务返回的 JSON dict
    """
    import urllib.error
    import urllib.request

    pdf_bytes = pdf_path.read_bytes()
    pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")

    payload = json.dumps(
        {
            "pdf_base64": pdf_b64,
            "output_format": "markdown",
        }
    ).encode("utf-8")

    url = modal_url.rstrip("/") + "/convert"
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"连接失败: {e.reason}") from e


def save_output(markdown: str, pdf_path: Path) -> Path:
    """将 markdown 保存到 output/ 目录。"""
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    stem = pdf_path.stem
    out_file = output_dir / f"marker_{stem}.md"
    out_file.write_text(markdown, encoding="utf-8")
    return out_file


def print_stats(pdf_path: Path, result: dict, out_file: Path, total_elapsed: float):
    """打印转换统计信息。"""
    page_count = result.get("page_count", "未知")
    server_elapsed = result.get("elapsed_seconds", "未知")
    md_size = len(result.get("markdown", ""))

    print("\n✅ 转换成功")
    print(f"  文件：{pdf_path.name}")
    print(f"  页数：{page_count}")
    print(f"  服务端耗时：{server_elapsed} 秒")
    print(f"  总耗时（含网络）：{total_elapsed:.1f} 秒")
    print(f"  Markdown 大小：{md_size:,} 字符")
    print(f"  输出：{out_file}")


def main():
    load_env()

    parser = argparse.ArgumentParser(
        description="验证 Modal marker 服务的 PDF→Markdown 转换"
    )
    parser.add_argument("--pdf", required=True, help="本地 PDF 文件路径")
    parser.add_argument(
        "--url",
        default=os.environ.get("MARKER_MODAL_URL", ""),
        help="Modal 服务 URL（也可通过 MARKER_MODAL_URL 环境变量设置）",
    )
    args = parser.parse_args()

    # 校验参数
    if not args.url:
        print(
            "❌ 错误：请通过 --url 参数或 MARKER_MODAL_URL 环境变量指定 Modal 服务 URL",
            file=sys.stderr,
        )
        sys.exit(1)

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"❌ 错误：PDF 文件不存在: {pdf_path}", file=sys.stderr)
        sys.exit(1)
    if pdf_path.suffix.lower() != ".pdf":
        print(f"❌ 错误：文件必须是 PDF 格式: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    file_size_mb = pdf_path.stat().st_size / 1024 / 1024
    print(f"📄 准备转换：{pdf_path.name}（{file_size_mb:.1f} MB）")
    print(f"🌐 Modal URL：{args.url}")
    print("⏳ 上传并转换中（冷启动可能需要 30-60 秒）...")

    start = time.time()
    try:
        result = convert_pdf(args.url, pdf_path)
    except RuntimeError as e:
        print(f"\n❌ 转换失败: {e}", file=sys.stderr)
        sys.exit(1)
    total_elapsed = time.time() - start

    if not result.get("success"):
        error = result.get("error", "未知错误")
        print(f"\n❌ 服务返回错误: {error}", file=sys.stderr)
        sys.exit(1)

    markdown = result.get("markdown", "")
    if not markdown:
        print("\n⚠️  警告：服务返回了空 Markdown", file=sys.stderr)

    out_file = save_output(markdown, pdf_path)
    print_stats(pdf_path, result, out_file, total_elapsed)

    # 打印前 500 字符供快速预览
    print("\n── Markdown 预览（前 500 字符）──")
    print(markdown[:500])
    if len(markdown) > 500:
        print(f"\n... （共 {len(markdown):,} 字符，完整内容见 {out_file}）")


if __name__ == "__main__":
    main()
