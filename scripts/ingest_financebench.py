#!/usr/bin/env python
"""批量入库 FinanceBench PDF 文档。

使用方式：
    python scripts/ingest_financebench.py --limit 10

流程：
1. 加载 FinanceBenchDataset
2. 获取唯一文档列表（doc_name）
3. 下载 PDF（如果需要）
4. 对每个文档调用 DocumentIngestion.ingest_pdf()
5. 记录成功/失败统计
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# 确保可以导入 pageindex_rag
_BASE_DIR = Path(__file__).parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from pageindex_rag.benchmark.financebench import FinanceBenchDataset
from pageindex_rag.ingestion.ingest import DocumentIngestion
from pageindex_rag.storage.models import Base
from pageindex_rag.storage.document_store import DocumentStore
from pageindex_rag.search.semantic_search import SemanticSearcher
from pageindex_rag.config import get_config


# FinanceBench PDF 下载基础 URL
_FINANCEBENCH_PDF_BASE = "https://github.com/PatronusAI/FinanceBench/raw/main/pdfs"


def get_pdf_path(doc_name: str, pdf_dir: str) -> str:
    """获取 PDF 文件路径。"""
    return os.path.join(pdf_dir, f"{doc_name}.pdf")


async def download_pdf(url: str, dest_path: str) -> bool:
    """下载 PDF 文件到指定路径。"""
    import httpx

    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            with open(dest_path, "wb") as f:
                f.write(response.content)
        print(f"  下载成功: {dest_path}")
        return True
    except Exception as e:
        print(f"  下载失败: {url} - {e}")
        return False


async def ingest_document(
    ingestion: DocumentIngestion,
    doc_info: dict,
    pdf_dir: str,
    force_download: bool = False,
) -> tuple[bool, str]:
    """入库单个文档。返回 (成功与否, 消息)。"""
    doc_name = doc_info["doc_name"]
    pdf_path = os.path.join(pdf_dir, f"{doc_name}.pdf")

    # 检查文件是否存在
    if not os.path.exists(pdf_path):
        if force_download:
            # 优先使用数据集中的 doc_link
            url = doc_info.get("doc_link") or f"{_FINANCEBENCH_PDF_BASE}/{doc_name}"
            print(f"  下载中: {url}")
            success = await download_pdf(url, pdf_path)
            if not success:
                return False, f"下载失败: {doc_name}"
        else:
            return False, f"文件不存在: {pdf_path}"

    # 构造元数据
    metadata = {
        "company": doc_info.get("company", ""),
        "fiscal_year": doc_info.get("fiscal_year", ""),
        "filing_type": doc_info.get("filing_type", ""),
    }

    try:
        doc_id = await ingestion.ingest_pdf(pdf_path, metadata=metadata)
        return True, f"成功入库: {doc_name} -> {doc_id}"
    except Exception as e:
        return False, f"入库失败: {doc_name} - {e}"


async def main():
    parser = argparse.ArgumentParser(description="批量入库 FinanceBench PDF 文档")
    parser.add_argument(
        "--limit", type=int, default=None, help="限制入库文档数量"
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default="./financebench_pdfs",
        help="PDF 文件存储目录",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="自动下载不存在的 PDF 文件",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="跳过已存在于数据库中的文档",
    )
    args = parser.parse_args()

    # 加载配置
    config = get_config()

    # 初始化数据库
    engine = create_engine(config.database_url)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)

    # 初始化组件
    document_store = DocumentStore(SessionLocal)
    semantic_searcher = SemanticSearcher(config)
    ingestion = DocumentIngestion(document_store, semantic_searcher, config)

    # 加载数据集
    print("加载 FinanceBench 数据集...")
    dataset = FinanceBenchDataset()
    unique_docs = dataset.get_unique_docs()

    if args.limit:
        unique_docs = unique_docs[: args.limit]

    print(f"共 {len(unique_docs)} 个唯一文档待处理")

    # 获取已入库的文档列表
    existing_docs = set()
    if args.skip_existing:
        for doc in document_store.list():
            existing_docs.add(doc["doc_name"])
        print(f"数据库中已有 {len(existing_docs)} 个文档")

    # 统计
    success_count = 0
    fail_count = 0
    skipped_count = 0

    # 批量入库
    for doc_info in unique_docs:
        doc_name = doc_info["doc_name"]

        # 跳过已存在的文档
        if args.skip_existing and doc_name in existing_docs:
            print(f"跳过已存在: {doc_name}")
            skipped_count += 1
            continue

        print(f"处理: {doc_name}")
        success, message = await ingest_document(
            ingestion, doc_info, args.pdf_dir, args.force_download
        )

        if success:
            print(f"  {message}")
            success_count += 1
        else:
            print(f"  {message}")
            fail_count += 1

    # 输出统计
    print("\n" + "=" * 50)
    print("入库完成")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  跳过: {skipped_count}")
    print(f"  总计: {len(unique_docs)}")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
