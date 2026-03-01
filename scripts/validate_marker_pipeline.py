"""
marker 端到端验证脚本 (Issue #42)

完整流程：
    PDF → marker(Modal) → Markdown → md_to_tree() → 树结构 JSON → 与黄金标准对比

用法：
    # 方式 1：使用 Modal 服务（需先部署）
    MARKER_MODAL_URL=https://xxx.modal.run \
        python scripts/validate_marker_pipeline.py \
        --pdf tests/fixtures/3M_2018_10K.pdf \
        --reference tests/fixtures/3M_2018_10K_tree.json

    # 方式 2：使用本地已有的 Markdown 文件（跳过 Modal 转换步骤）
    python scripts/validate_marker_pipeline.py \
        --md output/marker_3M_2018_10K.md \
        --reference tests/fixtures/3M_2018_10K_tree.json
"""

import argparse
import base64
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# 确保项目根目录在 path 中
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── 步骤 1：PDF → Markdown（via Modal）───────────────────────────────────────

def convert_pdf_via_modal(modal_url: str, pdf_path: Path) -> tuple[str, dict]:
    """
    调用 Modal marker 服务将 PDF 转为 Markdown。

    Returns:
        (markdown_text, stats_dict)
    """
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


# ── 步骤 2：Markdown → 树结构（via md_to_tree）───────────────────────────────

def markdown_to_tree(markdown: str, doc_name: str) -> list:
    """
    将 Markdown 文本写入临时文件，调用 md_to_tree() 生成树结构。

    Returns:
        树结构列表（与 3M_2018_10K_tree.json 中 structure 字段格式一致）
    """
    from pageindex_core.page_index_md import md_to_tree

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(markdown)
        tmp_path = tmp.name

    try:
        result = md_to_tree(
            tmp_path,
            if_thinning=False,
            if_add_node_id="yes",
        )
    finally:
        os.unlink(tmp_path)

    # md_to_tree 返回格式可能是 dict 或 list，统一取 structure
    if isinstance(result, dict):
        return result.get("structure", result.get("nodes", [result]))
    return result  # 已经是 list


# ── 步骤 3：与黄金标准对比 ────────────────────────────────────────────────────

def count_nodes(nodes: list) -> int:
    """递归统计树中所有节点数。"""
    total = len(nodes)
    for n in nodes:
        total += count_nodes(n.get("nodes", []))
    return total


def collect_titles(nodes: list, depth: int = 0) -> list[tuple[int, str]]:
    """递归收集所有节点标题，返回 [(depth, title), ...]。"""
    result = []
    for n in nodes:
        result.append((depth, n.get("title", "").strip()))
        result.extend(collect_titles(n.get("nodes", []), depth + 1))
    return result


def compare_trees(generated: list, reference: list) -> dict:
    """
    对比生成树与参考树的结构质量。

    返回报告 dict，包含：
    - node_count：节点数对比
    - top_level_count：顶层节点数对比
    - title_overlap：顶层标题重合率
    - depth_distribution：各深度节点数分布对比
    """
    gen_total = count_nodes(generated)
    ref_total = count_nodes(reference)

    gen_top = [n.get("title", "").strip() for n in generated]
    ref_top = [n.get("title", "").strip() for n in reference]

    # 标题重合（忽略大小写，模糊匹配：任意一方包含另一方）
    matched = 0
    for gt in gen_top:
        for rt in ref_top:
            if gt.lower() in rt.lower() or rt.lower() in gt.lower():
                matched += 1
                break

    overlap_rate = round(matched / max(len(ref_top), 1) * 100, 1)

    # 深度分布
    gen_titles = collect_titles(generated)
    ref_titles = collect_titles(reference)
    gen_depth_dist = {}
    ref_depth_dist = {}
    for d, _ in gen_titles:
        gen_depth_dist[d] = gen_depth_dist.get(d, 0) + 1
    for d, _ in ref_titles:
        ref_depth_dist[d] = ref_depth_dist.get(d, 0) + 1

    return {
        "node_count": {"generated": gen_total, "reference": ref_total},
        "top_level_count": {"generated": len(gen_top), "reference": len(ref_top)},
        "title_overlap_rate": f"{overlap_rate}%",
        "matched_top_titles": matched,
        "depth_distribution": {
            "generated": gen_depth_dist,
            "reference": ref_depth_dist,
        },
        "generated_top_titles": gen_top,
        "reference_top_titles": ref_top,
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
        description="验证 marker Pipeline：PDF/MD → 树结构 → 与黄金标准对比"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf", help="源 PDF 文件路径（使用 Modal 转换）")
    src.add_argument("--md", help="已有 Markdown 文件路径（跳过 Modal 步骤）")

    parser.add_argument(
        "--reference", required=True,
        help="黄金标准树结构 JSON 路径（如 tests/fixtures/3M_2018_10K_tree.json）"
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("MARKER_MODAL_URL", ""),
        help="Modal 服务 URL（--pdf 模式必须提供）"
    )
    parser.add_argument(
        "--output", default="",
        help="保存生成的树结构 JSON（可选，默认 output/marker_tree_<name>.json）"
    )
    args = parser.parse_args()

    ref_path = Path(args.reference)
    if not ref_path.exists():
        print(f"❌ 参考文件不存在: {ref_path}", file=sys.stderr)
        sys.exit(1)

    # ── 加载黄金标准 ──
    with open(ref_path, encoding="utf-8") as f:
        ref_data = json.load(f)
    reference_tree = ref_data.get("structure", ref_data)

    # ── 步骤 1：获取 Markdown ──
    stats = {}
    if args.pdf:
        if not args.url:
            print("❌ --pdf 模式需要提供 --url 或设置 MARKER_MODAL_URL", file=sys.stderr)
            sys.exit(1)
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"❌ PDF 文件不存在: {pdf_path}", file=sys.stderr)
            sys.exit(1)

        print(f"\n📄 步骤 1/3：PDF → Markdown（via Modal）")
        print(f"   文件：{pdf_path.name}（{pdf_path.stat().st_size / 1024:.0f} KB）")
        print(f"   URL：{args.url}")
        print("   ⏳ 转换中（冷启动约 30-60s）...")

        try:
            markdown, stats = convert_pdf_via_modal(args.url, pdf_path)
        except RuntimeError as e:
            print(f"   ❌ 转换失败: {e}", file=sys.stderr)
            sys.exit(1)

        # 保存 markdown
        output_dir = PROJECT_ROOT / "output"
        output_dir.mkdir(exist_ok=True)
        md_file = output_dir / f"marker_{pdf_path.stem}.md"
        md_file.write_text(markdown, encoding="utf-8")

        print(f"   ✅ 转换成功")
        print(f"      页数：{stats['page_count']}")
        print(f"      服务端耗时：{stats['server_elapsed']}s，总耗时：{stats['total_elapsed']}s")
        print(f"      Markdown：{stats['md_chars']:,} 字符，已保存到 {md_file}")
        doc_name = pdf_path.stem

    else:
        md_path = Path(args.md)
        if not md_path.exists():
            print(f"❌ Markdown 文件不存在: {md_path}", file=sys.stderr)
            sys.exit(1)
        print(f"\n📝 步骤 1/3：加载已有 Markdown 文件")
        print(f"   文件：{md_path}")
        markdown = md_path.read_text(encoding="utf-8")
        print(f"   字符数：{len(markdown):,}")
        doc_name = md_path.stem.replace("marker_", "")

    # ── 步骤 2：Markdown → 树结构 ──
    print(f"\n🌳 步骤 2/3：Markdown → 树结构（md_to_tree）")
    print("   ⏳ 生成中...")
    t0 = time.time()
    try:
        generated_tree = markdown_to_tree(markdown, doc_name)
    except Exception as e:
        print(f"   ❌ md_to_tree 失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    tree_elapsed = round(time.time() - t0, 2)
    print(f"   ✅ 生成成功，耗时 {tree_elapsed}s，顶层节点：{len(generated_tree)}")

    # 保存生成的树结构
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    out_json = Path(args.output) if args.output else output_dir / f"marker_tree_{doc_name}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"doc_name": doc_name, "structure": generated_tree}, f, ensure_ascii=False, indent=2)
    print(f"   已保存到：{out_json}")

    # ── 步骤 3：对比 ──
    print(f"\n📊 步骤 3/3：与黄金标准对比")
    report = compare_trees(generated_tree, reference_tree)

    gen_nodes = report["node_count"]["generated"]
    ref_nodes = report["node_count"]["reference"]
    node_diff = gen_nodes - ref_nodes
    node_diff_pct = round(abs(node_diff) / max(ref_nodes, 1) * 100, 1)

    print(f"\n{'='*55}")
    print(f"{'对比项':<25} {'生成结果':>12} {'黄金标准':>12}")
    print(f"{'-'*55}")
    print(f"{'总节点数':<25} {gen_nodes:>12} {ref_nodes:>12}")
    print(f"{'顶层节点数':<25} {report['top_level_count']['generated']:>12} {report['top_level_count']['reference']:>12}")
    print(f"{'顶层标题重合率':<25} {report['title_overlap_rate']:>12} {'100%':>12}")
    print(f"{'节点数偏差':<25} {f'{node_diff:+d} ({node_diff_pct}%)':>12}")
    print(f"{'='*55}")

    print(f"\n生成树顶层标题：")
    for t in report["generated_top_titles"]:
        print(f"  · {t}")

    print(f"\n参考树顶层标题：")
    for t in report["reference_top_titles"]:
        print(f"  · {t}")

    print(f"\n深度分布（生成 vs 参考）：")
    all_depths = sorted(
        set(report["depth_distribution"]["generated"]) |
        set(report["depth_distribution"]["reference"])
    )
    for d in all_depths:
        gen_d = report["depth_distribution"]["generated"].get(d, 0)
        ref_d = report["depth_distribution"]["reference"].get(d, 0)
        print(f"  深度 {d}：生成 {gen_d:3d}  参考 {ref_d:3d}")

    # 保存完整报告
    report_file = output_dir / f"marker_comparison_{doc_name}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "doc_name": doc_name,
            "conversion_stats": stats,
            "tree_generation_elapsed": tree_elapsed,
            "comparison": report,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n📄 完整对比报告：{report_file}")

    # 质量评估
    overlap = float(report["title_overlap_rate"].rstrip("%"))
    print(f"\n{'✅ 质量评估：PASS' if overlap >= 60 else '⚠️  质量评估：需人工复核'}")
    print(f"   顶层标题重合率 {overlap}%（阈值 60%）")


if __name__ == "__main__":
    main()
