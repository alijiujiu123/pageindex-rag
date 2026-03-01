"""
marker 端到端验证脚本 (Issue #42)

验证流程：
    PDF → marker(Modal) → Markdown → 提取标题层级 → 与黄金标准树结构对比

说明：
    本脚本只验证 PDF→MD 这一步的质量，不调用 LLM/md_to_tree()。
    通过对比 Markdown 标题（#/##/###）与黄金标准 JSON 中的节点 title，
    判断 marker 输出的结构是否完整、可供后续 md_to_tree() 使用。

用法：
    # 方式 1：使用 Modal 服务（需先部署）
    MARKER_MODAL_URL=https://xxx.modal.run \\
        python scripts/validate_marker_pipeline.py \\
        --pdf tests/fixtures/3M_2018_10K.pdf \\
        --reference tests/fixtures/3M_2018_10K_tree.json

    # 方式 2：使用本地已有的 Markdown 文件（跳过 Modal 步骤）
    python scripts/validate_marker_pipeline.py \\
        --md output/marker_3M_2018_10K.md \\
        --reference tests/fixtures/3M_2018_10K_tree.json
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── 步骤 1：PDF → Markdown（via Modal）───────────────────────────────────────

def convert_pdf_via_modal(modal_url: str, pdf_path: Path) -> tuple[str, dict]:
    import urllib.error
    import urllib.request

    pdf_b64 = base64.b64encode(pdf_path.read_bytes()).decode()
    payload = json.dumps({"pdf_base64": pdf_b64, "output_format": "markdown"}).encode()

    url = modal_url.rstrip("/") + "/convert"
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()}") from e

    elapsed = round(time.time() - start, 2)
    if not result.get("success"):
        raise RuntimeError(f"服务返回错误: {result.get('error')}")

    stats = {
        "page_count": result.get("page_count", 0),
        "server_elapsed": result.get("elapsed_seconds", 0),
        "total_elapsed": elapsed,
        "md_chars": len(result.get("markdown", "")),
    }
    return result["markdown"], stats


# ── 步骤 2：从 Markdown 提取标题层级（纯文本，无 LLM）─────────────────────────

def extract_md_headings(markdown: str) -> list[dict]:
    """
    从 Markdown 提取所有标题，返回 [{"level": 1, "title": "..."}]。
    level 1 = #, level 2 = ##, level 3 = ### ...
    """
    headings = []
    for line in markdown.splitlines():
        m = re.match(r"^(#{1,6})\s+(.+)", line)
        if m:
            headings.append({
                "level": len(m.group(1)),
                "title": m.group(2).strip(),
            })
    return headings


# ── 步骤 3：从黄金标准 JSON 提取节点标题 ─────────────────────────────────────

def extract_tree_titles(nodes: list, depth: int = 0) -> list[dict]:
    """递归提取树结构中所有节点标题。"""
    result = []
    for n in nodes:
        result.append({"depth": depth, "title": n.get("title", "").strip()})
        result.extend(extract_tree_titles(n.get("nodes", []), depth + 1))
    return result


# ── 步骤 4：对比分析 ──────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """标准化标题：小写、去多余空格。"""
    return re.sub(r"\s+", " ", text.lower().strip())


def titles_match(md_title: str, ref_title: str) -> bool:
    """模糊匹配：任意一方包含另一方（忽略大小写）。"""
    a, b = normalize(md_title), normalize(ref_title)
    return a in b or b in a


def compare(md_headings: list[dict], ref_titles: list[dict]) -> dict:
    """
    对比 MD 标题与参考树节点 titles。

    - heading_counts：各级标题数量
    - ref_title_coverage：参考树中有多少 title 能在 MD 标题里找到匹配
    - unmatched_ref_titles：未匹配的参考节点标题（可能是 marker 漏了）
    """
    md_title_list = [h["title"] for h in md_headings]
    ref_title_list = [t["title"] for t in ref_titles]

    matched, unmatched = [], []
    for rt in ref_title_list:
        if any(titles_match(mt, rt) for mt in md_title_list):
            matched.append(rt)
        else:
            unmatched.append(rt)

    coverage = round(len(matched) / max(len(ref_title_list), 1) * 100, 1)

    heading_counts = {}
    for h in md_headings:
        lv = h["level"]
        heading_counts[lv] = heading_counts.get(lv, 0) + 1

    return {
        "md_total_headings": len(md_headings),
        "ref_total_nodes": len(ref_title_list),
        "heading_level_distribution": heading_counts,
        "ref_title_coverage": f"{coverage}%",
        "matched_count": len(matched),
        "unmatched_count": len(unmatched),
        "matched_titles": matched,
        "unmatched_ref_titles": unmatched,
    }


# ── 主流程 ────────────────────────────────────────────────────────────────────

def load_env():
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())


def main():
    load_env()

    parser = argparse.ArgumentParser(
        description="验证 marker PDF→MD 转换质量（纯结构对比，不调用 LLM）"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf", help="源 PDF 文件路径（使用 Modal 转换）")
    src.add_argument("--md", help="已有 Markdown 文件路径（跳过 Modal 步骤）")

    parser.add_argument(
        "--reference", required=True,
        help="黄金标准树结构 JSON（如 tests/fixtures/3M_2018_10K_tree.json）"
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("MARKER_MODAL_URL", ""),
        help="Modal 服务 URL（--pdf 模式必须提供）"
    )
    args = parser.parse_args()

    # 加载黄金标准
    ref_path = Path(args.reference)
    if not ref_path.exists():
        print(f"❌ 参考文件不存在: {ref_path}", file=sys.stderr)
        sys.exit(1)
    with open(ref_path, encoding="utf-8") as f:
        ref_data = json.load(f)
    reference_tree = ref_data.get("structure", ref_data)
    ref_titles = extract_tree_titles(reference_tree)

    # 步骤 1：获取 Markdown
    stats = {}
    if args.pdf:
        if not args.url:
            print("❌ --pdf 模式需要提供 --url 或设置 MARKER_MODAL_URL", file=sys.stderr)
            sys.exit(1)
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"❌ PDF 文件不存在: {pdf_path}", file=sys.stderr)
            sys.exit(1)

        print(f"\n📄 步骤 1/3：PDF → Markdown（via Modal marker）")
        print(f"   文件：{pdf_path.name}（{pdf_path.stat().st_size / 1024:.0f} KB）")
        print(f"   URL：{args.url}")
        print("   ⏳ 转换中（冷启动约 30-60s）...")

        try:
            markdown, stats = convert_pdf_via_modal(args.url, pdf_path)
        except RuntimeError as e:
            print(f"   ❌ 转换失败: {e}", file=sys.stderr)
            sys.exit(1)

        # 保存 Markdown
        output_dir = PROJECT_ROOT / "output"
        output_dir.mkdir(exist_ok=True)
        md_file = output_dir / f"marker_{pdf_path.stem}.md"
        md_file.write_text(markdown, encoding="utf-8")

        print(f"   ✅ 转换成功")
        print(f"      页数：{stats['page_count']}，"
              f"服务端耗时：{stats['server_elapsed']}s，总耗时：{stats['total_elapsed']}s")
        print(f"      Markdown：{stats['md_chars']:,} 字符 → {md_file}")
        doc_name = pdf_path.stem

    else:
        md_path = Path(args.md)
        if not md_path.exists():
            print(f"❌ Markdown 文件不存在: {md_path}", file=sys.stderr)
            sys.exit(1)
        markdown = md_path.read_text(encoding="utf-8")
        doc_name = md_path.stem.replace("marker_", "")
        print(f"\n📝 步骤 1/3：加载 Markdown 文件")
        print(f"   {md_path}（{len(markdown):,} 字符）")

    # 步骤 2：提取 MD 标题
    print(f"\n🔍 步骤 2/3：提取 Markdown 标题层级")
    md_headings = extract_md_headings(markdown)
    print(f"   共提取 {len(md_headings)} 个标题")

    level_dist = {}
    for h in md_headings:
        level_dist[h["level"]] = level_dist.get(h["level"], 0) + 1
    for lv in sorted(level_dist):
        print(f"   {'#' * lv} ：{level_dist[lv]} 个")

    # 步骤 3：对比
    print(f"\n📊 步骤 3/3：与黄金标准对比")
    print(f"   参考树节点总数：{len(ref_titles)}")
    report = compare(md_headings, ref_titles)
    coverage = float(report["ref_title_coverage"].rstrip("%"))

    print(f"\n{'='*55}")
    print(f"  MD 标题总数：{report['md_total_headings']}")
    print(f"  参考节点总数：{report['ref_total_nodes']}")
    print(f"  参考标题覆盖率：{report['ref_title_coverage']}  "
          f"({report['matched_count']}/{report['ref_total_nodes']})")
    print(f"{'='*55}")

    if report["unmatched_ref_titles"]:
        print(f"\n⚠️  未在 MD 中找到的参考标题（{report['unmatched_count']} 个）：")
        for t in report["unmatched_ref_titles"]:
            print(f"   · {t}")

    # 保存报告
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    report_file = output_dir / f"marker_validation_{doc_name}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "doc_name": doc_name,
            "conversion_stats": stats,
            "comparison": report,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n📄 完整报告：{report_file}")

    # 质量结论
    print(f"\n{'✅ PASS' if coverage >= 60 else '⚠️  需人工复核'}  "
          f"参考标题覆盖率 {coverage}%（阈值 60%）")
    if coverage >= 60:
        print("   marker 输出的 Markdown 标题结构完整，可供 md_to_tree() 使用。")
    else:
        print("   marker 标题提取不足，请检查 output/marker_<name>.md 文件。")


if __name__ == "__main__":
    main()
