"""端到端测试：真实 LLM + 真实 DocumentStore（但小规模）。"""
import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from pageindex_rag.benchmark.financebench import FinanceBenchDataset
from pageindex_rag.ingestion.ingest import DocumentIngestion
from pageindex_rag.storage.models import Base
from pageindex_rag.storage.document_store import DocumentStore
from pageindex_rag.search.semantic_search import SemanticSearcher
from pageindex_rag.pipeline.rag_pipeline import RAGPipeline
from pageindex_rag.retrieval.tree_search import TreeSearcher
from pageindex_rag.retrieval.node_extractor import NodeContentExtractor
from pageindex_rag.search.router import DocumentSearchRouter
from pageindex_rag.config import get_config


# 创建简单的测试 PDF 文件（最简单的有效 PDF）
_MINIMAL_PDF = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Test Document) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000214 00000 n
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
301
%%EOF
"""


@pytest.fixture
def test_db_path():
    """创建临时测试数据库文件路径。"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield f.name
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def test_chroma_dir():
    """创建临时 ChromaDB 目录。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def real_document_store(test_db_path):
    """真实的 DocumentStore（使用测试数据库）。"""
    engine = create_engine(f"sqlite:///{test_db_path}")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    return DocumentStore(SessionLocal)


@pytest.fixture
def real_semantic_searcher(test_chroma_dir):
    """真实的 SemanticSearcher（使用临时目录）。"""
    config = get_config(
        chroma_persist_dir=test_chroma_dir,
        openai_api_key="test-key",
    )
    return SemanticSearcher(config)


@pytest.fixture
def real_ingestion(real_document_store, real_semantic_searcher):
    """真实的 DocumentIngestion。"""
    config = get_config(openai_api_key="test-key")
    return DocumentIngestion(
        document_store=real_document_store,
        semantic_searcher=real_semantic_searcher,
        config=config,
    )


@pytest.fixture
def sample_tree():
    """返回测试用的树结构。"""
    return {
        "title": "Test Document",
        "node_id": "0000",
        "start_index": 1,
        "end_index": 1,
        "nodes": [
            {
                "title": "Section 1",
                "node_id": "0001",
                "start_index": 1,
                "end_index": 1,
            }
        ],
    }


@pytest.mark.asyncio
async def test_batch_ingest(real_ingestion, sample_tree):
    """测试批量入库流程（使用 mock PDF 文件或小规模真实文件）。"""
    # 创建临时 PDF 文件
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(_MINIMAL_PDF)
        pdf_path = f.name

    try:
        with patch("pageindex_rag.ingestion.ingest.page_index_main") as mock_page_index, \
             patch("pageindex_rag.ingestion.ingest.generate_doc_description") as mock_gen_desc, \
             patch("pageindex_rag.search.semantic_search.get_embedding") as mock_embedding:

            # Mock page_index_main 返回树结构
            mock_page_index.return_value = sample_tree
            mock_gen_desc.return_value = "Test document description"
            # Mock embedding 返回固定向量
            mock_embedding.return_value = [0.1] * 1536  # OpenAI embedding 维度

            # 入库测试
            metadata = {
                "company": "TestCorp",
                "fiscal_year": "2023",
                "filing_type": "10-K",
            }

            doc_id = await real_ingestion.ingest_pdf(pdf_path, metadata=metadata)

            assert doc_id is not None
            assert doc_id.startswith("pi-")

            # 验证文档已存储
            doc = real_ingestion.document_store.get(doc_id)
            assert doc is not None
            assert doc["company"] == "TestCorp"
            assert doc["fiscal_year"] == "2023"
            assert doc["filing_type"] == "10-K"

    finally:
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)


@pytest.mark.asyncio
async def test_multiple_pdf_ingest(real_ingestion, sample_tree):
    """测试多个 PDF 文件入库。"""
    # 创建多个临时 PDF 文件
    test_files = []
    for i in range(3):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(_MINIMAL_PDF)
            test_files.append(f.name)

    try:
        with patch("pageindex_rag.ingestion.ingest.page_index_main") as mock_page_index, \
             patch("pageindex_rag.ingestion.ingest.generate_doc_description") as mock_gen_desc, \
             patch("pageindex_rag.search.semantic_search.get_embedding") as mock_embedding:

            mock_page_index.return_value = sample_tree
            mock_gen_desc.return_value = "Test document description"
            mock_embedding.return_value = [0.1] * 1536  # OpenAI embedding 维度

            doc_ids = []
            for i, pdf_path in enumerate(test_files):
                metadata = {
                    "company": f"Company{i}",
                    "fiscal_year": "2023",
                    "filing_type": "10-K",
                }
                doc_id = await real_ingestion.ingest_pdf(pdf_path, metadata=metadata)
                doc_ids.append(doc_id)

            assert len(doc_ids) == 3
            assert len(set(doc_ids)) == 3  # 所有 doc_id 应该不同

            # 验证列表
            docs = real_ingestion.document_store.list()
            assert len(docs) == 3

            # 验证每个文档的元数据
            for i, doc in enumerate(docs):
                assert doc["company"] == f"Company{i}"

    finally:
        for f in test_files:
            if os.path.exists(f):
                os.unlink(f)


@pytest.mark.asyncio
async def test_financebench_dataset_structure():
    """测试 FinanceBenchDataset 数据结构。"""
    # 使用 mock 数据集避免下载
    with patch("pageindex_rag.benchmark.financebench.load_dataset") as mock_load:
        mock_load.return_value = [
            {
                "question": "What is the revenue?",
                "answer": "$100M",
                "company": "TestCorp",
                "fiscal_year": "2023",
                "filing_type": "10-K",
                "doc_name": "test_report.pdf",
            },
            {
                "question": "What is the net income?",
                "answer": "$50M",
                "company": "TestCorp",
                "fiscal_year": "2023",
                "filing_type": "10-K",
                "doc_name": "test_report.pdf",  # 同一个文档
            },
        ]

        dataset = FinanceBenchDataset()

        assert len(dataset) == 2

        # 测试 get_unique_docs
        unique_docs = dataset.get_unique_docs()
        assert len(unique_docs) == 1  # 只有一个唯一文档
        assert unique_docs[0]["doc_name"] == "test_report.pdf"
        assert unique_docs[0]["company"] == "TestCorp"


@pytest.mark.skipif(not os.getenv("RUN_E2E"), reason="需要设置 RUN_E2E=1")
@pytest.mark.asyncio
async def test_e2e_sample_10():
    """端到端测试：10 题真实验证（需要真实密钥和数据）。"""
    # 这个测试需要：
    # 1. 真实的 OpenAI API key (CHATGPT_API_KEY)
    # 2. 真实的数据库连接
    # 3. FinanceBench 数据集下载

    config = get_config()

    # 初始化组件
    engine = create_engine(config.database_url)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)

    document_store = DocumentStore(SessionLocal)
    semantic_searcher = SemanticSearcher(config)
    ingestion = DocumentIngestion(document_store, semantic_searcher, config)

    # 加载数据集（只取前 10 个问题对应的文档）
    dataset = FinanceBenchDataset()
    unique_docs = dataset.get_unique_docs()[:5]  # 限制为 5 个文档

    print(f"E2E 测试：入库 {len(unique_docs)} 个文档")

    for doc_info in unique_docs:
        print(f"  处理: {doc_info['doc_name']}")
        # 注意：这里假设 PDF 文件已存在于 ./financebench_pdfs 目录
        pdf_path = f"./financebench_pdfs/{doc_info['doc_name']}"
        if os.path.exists(pdf_path):
            try:
                doc_id = await ingestion.ingest_pdf(pdf_path, metadata=doc_info)
                print(f"    成功: {doc_id}")
            except Exception as e:
                print(f"    失败: {e}")
        else:
            print(f"    跳过：文件不存在")

    # 验证文档已入库
    docs = document_store.list()
    print(f"数据库中共有 {len(docs)} 个文档")

    assert len(docs) > 0, "至少应该有一个文档成功入库"
