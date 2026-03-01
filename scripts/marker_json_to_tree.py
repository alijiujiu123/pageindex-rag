"""
marker JSON → PageIndex tree.json 适配器 (Issue #42)

将 marker output_format=json 输出的 JSONOutput 结构转换为 PageIndex tree.json 格式。

PageIndex tree.json 格式：
    {
        "doc_name": "xxx.pdf",
        "structure": [
            {
                "title": "PART I",
                "node_id": "0000",
                "start_index": 1,
                "end_index": 10,
                "nodes": [...]
            },
            ...
        ]
    }

转换规则：
    - title：解析 SectionHeader 的 html 字段，去除 HTML 标签
    - node_id：按遍历顺序 zfill(4)，全局计数
    - start_index：从 block id "/page/N/..." 解析页码（0-indexed → +1）
    - end_index：下一同级节点 start_index - 1；最后一个节点 = 总页数
    - nodes：递归提取 children 中的 SectionHeader 类型
    - summary：不生成（本阶段省略，后续可选补充）

用法（模块内）：
    from scripts.marker_json_to_tree import marker_json_to_pageindex_tree
    tree = marker_json_to_pageindex_tree(marker_json, doc_name="3M_2018_10K.pdf")
"""

import re
from typing import Optional


def _strip_html(html: str) -> str:
    """去除 HTML 标签，返回纯文本。"""
    text = re.sub(r"<[^>]+>", "", html)
    return text.strip()


def _parse_page_from_id(block_id: str) -> Optional[int]:
    """
    从 block id 解析页码（1-indexed）。
    例如 "/page/5/SectionHeader/0" → 6（0-indexed + 1）
    """
    m = re.match(r"^/page/(\d+)/", block_id)
    if m:
        return int(m.group(1)) + 1
    return None


def _collect_section_headers(blocks: list) -> list[dict]:
    """
    递归收集 marker JSON 中所有 SectionHeader 块（保留层级关系）。

    返回每个节点包含：
        {
            "html": str,
            "id": str,
            "children_headers": [...]   # 直接子级 SectionHeader
        }
    """
    result = []
    for block in blocks:
        block_type = block.get("block_type", "")
        children = block.get("children") or []

        if block_type == "SectionHeader":
            # 在当前 SectionHeader 的 children 中查找子级 SectionHeader
            child_headers = _collect_section_headers(children)
            result.append({
                "html": block.get("html", ""),
                "id": block.get("id", ""),
                "children_headers": child_headers,
            })
        elif block_type == "Page":
            # Page 层：继续递归查找其中的 SectionHeader
            result.extend(_collect_section_headers(children))
        else:
            # 其他类型（Text, Table, Figure 等）：继续递归子节点
            result.extend(_collect_section_headers(children))

    return result


_node_counter = 0


def _build_nodes(
    headers: list[dict],
    total_pages: int,
    next_sibling_start: Optional[int] = None,
) -> list[dict]:
    """
    递归将 headers 列表转换为 PageIndex 节点格式。

    Args:
        headers: _collect_section_headers 返回的列表
        total_pages: PDF 总页数（用于最后一个节点的 end_index）
        next_sibling_start: 下一同级节点的 start_index（用于计算 end_index）
    """
    global _node_counter
    nodes = []

    for i, header in enumerate(headers):
        # 计算 start_index
        start = _parse_page_from_id(header["id"])
        if start is None:
            start = 1  # 降级处理

        # 计算下一同级节点 start（用于 end_index 推算）
        if i + 1 < len(headers):
            next_start = _parse_page_from_id(headers[i + 1]["id"]) or (start + 1)
        else:
            # 最后一个同级节点：end 由上层传入的 next_sibling_start 决定
            next_start = next_sibling_start if next_sibling_start is not None else (total_pages + 1)

        # end_index：下一同级节点 start - 1
        end = next_start - 1

        # 父节点先分配 node_id，再递归子节点
        node_id = str(_node_counter).zfill(4)
        _node_counter += 1

        child_nodes = _build_nodes(
            header["children_headers"],
            total_pages,
            next_sibling_start=next_start,
        )

        nodes.append({
            "title": _strip_html(header["html"]),
            "node_id": node_id,
            "start_index": start,
            "end_index": max(end, start),  # 保证 end >= start
            "nodes": child_nodes,
        })

    return nodes


def marker_json_to_pageindex_tree(marker_json: dict, doc_name: str) -> dict:
    """
    将 marker JSONOutput 转换为 PageIndex tree.json 格式。

    Args:
        marker_json: marker output_format=json 的输出 dict
                     顶层字段：block_type="Document", children=[Page,...], metadata={pages:N, toc:[...]}
        doc_name: 文档名称（如 "3M_2018_10K.pdf"）

    Returns:
        {"doc_name": str, "structure": [...]}
    """
    global _node_counter
    _node_counter = 0  # 重置计数器（每次调用独立）

    # 获取总页数
    metadata = marker_json.get("metadata") or {}
    total_pages = metadata.get("pages", 0)

    # 提取顶层 children（Page 列表）
    children = marker_json.get("children") or []

    # 收集所有 SectionHeader（保留层级）
    all_headers = _collect_section_headers(children)

    # 构建 PageIndex 树节点
    structure = _build_nodes(all_headers, total_pages)

    return {
        "doc_name": doc_name,
        "structure": structure,
    }


# ── CLI 入口（独立使用）────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="将 marker JSON 输出转换为 PageIndex tree.json 格式"
    )
    parser.add_argument("--marker-json", required=True, help="marker 原始 JSON 文件路径")
    parser.add_argument("--output", help="输出 tree.json 文件路径（默认打印到 stdout）")
    parser.add_argument("--doc-name", help="文档名称（默认从输入文件名推断）")
    args = parser.parse_args()

    input_path = Path(args.marker_json)
    if not input_path.exists():
        print(f"❌ 文件不存在: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        marker_json = json.load(f)

    doc_name = args.doc_name or input_path.stem.replace("marker_raw_", "") + ".pdf"
    tree = marker_json_to_pageindex_tree(marker_json, doc_name)

    output_str = json.dumps(tree, ensure_ascii=False, indent=2)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_str, encoding="utf-8")
        print(f"✅ 转换完成：{out_path}")
        print(f"   顶层节点数：{len(tree['structure'])}")
    else:
        print(output_str)
