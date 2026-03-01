"""
marker JSON 路径端到端验证脚本 (Issue #42)

验证流程（无 LLM 调用）：
    PDF → marker(Modal, output_format=json) → marker_json → 适配转换 → tree.json → 对比参考树

两种运行模式：

  阶段 A（全流程，需 Modal 服务）：
    MARKER_MODAL_URL=https://xxx.modal.run \\
        python scripts/validate_marker_pipeline.py \\
        --pdf tests/fixtures/3M_2018_10K.pdf \\
        --reference tests/fixtures/3M_2018_10K_tree.json

  阶段 B（跳过 Modal，用已有 marker json 文件）：
    python scripts/validate_marker_pipeline.py \\
        --marker-json output/marker_raw_3M_2018_10K.json \\
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

from scripts.marker_json_to_tree import marker_json_to_pageindex_tree


# ── 步骤 1：PDF → marker JSON（via Modal）─────────────────────────────────────

def convert_pdf_to_marker_json(modal_url: str, pdf_path: Path) -> tuple[dict, dict]:
    """
    上传 PDF 到 Modal marker 服务，获取 marker JSON 输出。

    Returns:
        (marker_json, stats)
    """
    import urllib.error
    import urllib.request

    pdf_b64 = base64.b64encode(pdf_path.read_bytes()).decode()
    payload = json.dumps({
        "pdf_base64": pdf_b64,
        "output_format": "json",
    }).encode()

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
    except urllib.error.URLError as e:
        raise RuntimeError(f"连接失败: {e.reason}") from e

    elapsed = round(time.time() - start, 2)

    if not result.get("success"):
        raise RuntimeError(f"服务返回错误: {result.get('error')}")

    stats = {
        "page_count": result.get("page_count", 0),
        "server_elapsed": result.get("elapsed_seconds", 0),
        "total_elapsed": elapsed,
    }
    return result["marker_json"], stats


# ── 步骤 2：提取树节点标题（递归）──────────────────────────────────────────────

def extract_tree_titles(nodes: list, depth: int = 0) -> list[dict]:
    """递归提取树结构中所有节点标题。"""
    result = []
    for n in nodes:
        result.append({"depth": depth, "title": n.get("title", "").strip()})
        result.extend(extract_tree_titles(n.get("nodes", []), depth + 1))
    return result


# ── 步骤 3：对比分析 ───────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def titles_match(a: str, b: str) -> bool:
    na, nb = normalize(a), normalize(b)
    return na in nb or nb in na


def compare_trees(generated: list[dict], reference: list[dict]) -> dict:
    """
    对比生成树与参考树的标题覆盖率。

    Returns:
        {
            "generated_total": int,
            "reference_total": int,
            "coverage_pct": float,
            "matched_count": int,
            "unmatched_ref_titles": [str],
        }
    """
    gen_titles = [n["title"] for n in generated]
    ref_titles = [n["title"] for n in reference]

    matched, unmatched = [], []
    for rt in ref_titles:
        if any(titles_match(gt, rt) for gt in gen_titles):
            matched.append(rt)
        else:
            unmatched.append(rt)

    coverage = round(len(matched) / max(len(ref_titles), 1) * 100, 1)

    return {
        "generated_total": len(gen_titles),
        "reference_total": len(ref_titles),
        "coverage_pct": coverage,
        "matched_count": len(matched),
        "unmatched_ref_titles": unmatched,
    }


# ── 辅助 ──────────────────────────────────────────────────────────────────────

def load_env():
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())


def print_tree_titles(structure: list, indent: int = 0):
    """打印树结构标题供肉眼审查。"""
    for node in structure:
        print("  " * indent + f"· {node['title']}")
        if node.get("nodes"):
            print_tree_titles(node["nodes"], indent + 1)


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    load_env()

    parser = argparse.ArgumentParser(
        description="验证 marker JSON 路径 PDF→tree.json 转换质量（无 LLM 调用）"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf", help="源 PDF 文件路径（使用 Modal marker json 模式）")
    src.add_argument("--marker-json", help="已有 marker JSON 文件路径（跳过 Modal 步骤）")

    parser.add_argument(
        "--reference", required=True,
        help="参考树结构 JSON（如 tests/fixtures/3M_2018_10K_tree.json）"
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("MARKER_MODAL_URL", ""),
        help="Modal 服务 URL（--pdf 模式必须提供）"
    )
    args = parser.parse_args()

    # 加载参考树
    ref_path = Path(args.reference)
    if not ref_path.exists():
        print(f"❌ 参考文件不存在: {ref_path}", file=sys.stderr)
        sys.exit(1)
    with open(ref_path, encoding="utf-8") as f:
        ref_data = json.load(f)
    ref_structure = ref_data.get("structure", ref_data)
    ref_titles = extract_tree_titles(ref_structure)
    doc_name = ref_data.get("doc_name", ref_path.stem)

    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(exist_ok=True)

    # ── 阶段 A：PDF → marker JSON（via Modal）──────────────────────────────
    stats = {}
    if args.pdf:
        if not args.url:
            print("❌ --pdf 模式需要提供 --url 或设置 MARKER_MODAL_URL", file=sys.stderr)
            sys.exit(1)
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"❌ PDF 文件不存在: {pdf_path}", file=sys.stderr)
            sys.exit(1)

        print(f"\n📄 阶段 A-1：PDF → marker JSON（via Modal）")
        print(f"   文件：{pdf_path.name}（{pdf_path.stat().st_size / 1024:.0f} KB）")
        print(f"   URL：{args.url}")
        print("   ⏳ 转换中（冷启动约 30-60s）...")

        try:
            marker_json, stats = convert_pdf_to_marker_json(args.url, pdf_path)
        except RuntimeError as e:
            print(f"   ❌ 转换失败: {e}", file=sys.stderr)
            sys.exit(1)

        # 保存原始 marker JSON
        raw_file = output_dir / f"marker_raw_{pdf_path.stem}.json"
        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(marker_json, f, ensure_ascii=False, indent=2)

        print(f"   ✅ 转换成功")
        print(f"      页数：{stats['page_count']}，"
              f"服务端耗时：{stats['server_elapsed']}s，总耗时：{stats['total_elapsed']}s")
        print(f"      marker JSON 保存至：{raw_file}")
        stem = pdf_path.stem

    else:
        # ── 阶段 B：加载已有 marker JSON ───────────────────────────────────
        mj_path = Path(args.marker_json)
        if not mj_path.exists():
            print(f"❌ marker JSON 文件不存在: {mj_path}", file=sys.stderr)
            sys.exit(1)
        with open(mj_path, encoding="utf-8") as f:
            marker_json = json.load(f)
        stem = mj_path.stem.replace("marker_raw_", "")
        print(f"\n📂 阶段 A（跳过）：加载本地 marker JSON")
        print(f"   {mj_path}")

    # ── 阶段 A-2：marker JSON → PageIndex tree ──────────────────────────────
    print(f"\n🔧 阶段 A-2：marker JSON → PageIndex tree（适配转换）")
    tree = marker_json_to_pageindex_tree(marker_json, doc_name=doc_name)
    gen_structure = tree["structure"]

    # 保存生成的树结构
    tree_file = output_dir / f"marker_tree_{stem}.json"
    with open(tree_file, "w", encoding="utf-8") as f:
        json.dump(tree, f, ensure_ascii=False, indent=2)

    gen_titles = extract_tree_titles(gen_structure)
    print(f"   ✅ 生成树节点总数：{len(gen_titles)}（顶层：{len(gen_structure)}）")
    print(f"   tree.json 保存至：{tree_file}")

    # ── 阶段 B：对比报告 ────────────────────────────────────────────────────
    print(f"\n📊 阶段 B：与参考树对比")
    print(f"   参考树节点总数：{len(ref_titles)}")

    report = compare_trees(gen_titles, ref_titles)
    coverage = report["coverage_pct"]

    print(f"\n{'='*60}")
    print(f"  生成树节点数：{report['generated_total']}")
    print(f"  参考树节点数：{report['reference_total']}")
    print(f"  参考标题覆盖率：{coverage}%  ({report['matched_count']}/{report['reference_total']})")
    print(f"{'='*60}")

    if report["unmatched_ref_titles"]:
        print(f"\n⚠️  未覆盖的参考标题（{len(report['unmatched_ref_titles'])} 个）：")
        for t in report["unmatched_ref_titles"]:
            print(f"   · {t}")

    # ── 阶段 B（手动审查）：打印生成树标题 ──────────────────────────────────
    print(f"\n── 生成树标题列表（供肉眼审查）──")
    print_tree_titles(gen_structure)

    # 保存完整报告
    report_file = output_dir / f"marker_validation_{stem}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "doc_name": doc_name,
            "conversion_stats": stats,
            "comparison": report,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n📄 完整报告：{report_file}")

    # 结论
    threshold = 60.0
    passed = coverage >= threshold
    print(f"\n{'✅ PASS' if passed else '⚠️  WARN'}  "
          f"参考标题覆盖率 {coverage}%（阈值 {threshold}%）")
    if passed:
        print("   marker JSON 路径生成的树结构质量达标，可替代 PageIndex SDK。")
    else:
        print("   覆盖率不足，请检查 marker JSON 输出或调整适配规则。")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
