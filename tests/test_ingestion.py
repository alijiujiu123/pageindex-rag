import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# PageIndex SDK 实际返回的树结构（解码后）
SAMPLE_TREE = [
    {
        "title": "Test Document",
        "node_id": "0001",
        "start_index": 1,
        "end_index": 10,
        "text": "Sample text content...",
        "nodes": [
            {
                "title": "Chapter 1",
                "node_id": "0002",
                "start_index": 1,
                "end_index": 5,
                "text": "Chapter 1 content...",
                "nodes": []
            }
        ]
    }
]

# PageIndex SDK 实际返回格式：双重 JSON 编码的字符串列表
SAMPLE_TREE_SDK_FORMAT = [json.dumps(json.dumps(node)) for node in SAMPLE_TREE]

SAMPLE_MD_TREE = {
    "title": "Test Document",
    "node_id": "0001",
    "start_index": 1,
    "end_index": 10,
    "nodes": [
        {
            "title": "Chapter 1",
            "node_id": "0002",
            "start_index": 1,
            "end_index": 5,
            "nodes": []
        }
    ]
}


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.pageindex_api_key = "test-api-key"
    config.model = "gpt-4o-mini"
    return config


@pytest.fixture
def mock_document_store():
    store = MagicMock()
    store.create.return_value = "pi-test-550e8400"
    return store


@pytest.fixture
def mock_semantic_searcher():
    searcher = MagicMock()
    searcher.index_document.return_value = None
    return searcher


@pytest.fixture
def mock_pi_client():
    """Mock PageIndex SDK client"""
    client = MagicMock()
    client.submit_document.return_value = {"doc_id": "cloud-doc-123"}
    client.get_tree.return_value = {
        "status": "completed",
        "result": SAMPLE_TREE_SDK_FORMAT
    }
    client.delete_document.return_value = None
    return client


@pytest.mark.asyncio
async def test_ingest_pdf_uses_pageindex_sdk(mock_config, mock_document_store, mock_semantic_searcher, mock_pi_client):
    """ingest_pdf 应使用 PageIndex SDK 云服务处理 PDF"""
    from pageindex_rag.ingestion.ingest import DocumentIngestion

    with patch("pageindex_rag.ingestion.ingest.PageIndexClient") as mock_client_class, \
         patch("pageindex_rag.ingestion.ingest.generate_doc_description") as mock_gen_desc:

        mock_client_class.return_value = mock_pi_client
        mock_gen_desc.return_value = "这是一份测试PDF文档的描述。"

        ingestion = DocumentIngestion(
            document_store=mock_document_store,
            semantic_searcher=mock_semantic_searcher,
            config=mock_config
        )

        doc_id = await ingestion.ingest_pdf("/path/to/test.pdf")

        # PageIndex SDK submit_document 应被调用
        mock_pi_client.submit_document.assert_called_once_with("/path/to/test.pdf")
        # PageIndex SDK get_tree 应被调用
        mock_pi_client.get_tree.assert_called()
        # 验证返回的 doc_id
        assert doc_id == "pi-test-550e8400"


@pytest.mark.asyncio
async def test_ingest_pdf_generates_description(mock_config, mock_document_store, mock_semantic_searcher, mock_pi_client):
    """ingest_pdf 无 doc_description 时应调用 generate_doc_description"""
    from pageindex_rag.ingestion.ingest import DocumentIngestion

    with patch("pageindex_rag.ingestion.ingest.PageIndexClient") as mock_client_class, \
         patch("pageindex_rag.ingestion.ingest.generate_doc_description") as mock_gen_desc:

        mock_client_class.return_value = mock_pi_client
        mock_gen_desc.return_value = "这是一份测试PDF文档的描述。"

        ingestion = DocumentIngestion(
            document_store=mock_document_store,
            semantic_searcher=mock_semantic_searcher,
            config=mock_config
        )

        doc_id = await ingestion.ingest_pdf("/path/to/test.pdf")

        assert doc_id == "pi-test-550e8400"
        mock_gen_desc.assert_called_once()
        mock_document_store.create.assert_called_once_with(
            "/path/to/test.pdf",
            SAMPLE_TREE,
            {"doc_description": "这是一份测试PDF文档的描述。"}
        )
        mock_semantic_searcher.index_document.assert_called_once_with(
            "pi-test-550e8400", SAMPLE_TREE
        )


@pytest.mark.asyncio
async def test_ingest_pdf_no_api_key_raises(mock_document_store, mock_semantic_searcher):
    """无 PAGEINDEX_API_KEY 时应抛出 ValueError"""
    from pageindex_rag.ingestion.ingest import DocumentIngestion

    config_no_key = MagicMock()
    config_no_key.pageindex_api_key = ""

    with patch("pageindex_rag.ingestion.ingest.PageIndexClient") as mock_client_class:
        mock_client_class.return_value = None

        ingestion = DocumentIngestion(
            document_store=mock_document_store,
            semantic_searcher=mock_semantic_searcher,
            config=config_no_key
        )

        with pytest.raises(ValueError, match="PageIndex API key not configured"):
            await ingestion.ingest_pdf("/path/to/test.pdf")


@pytest.mark.asyncio
async def test_ingest_pdf_polls_until_completed(mock_config, mock_document_store, mock_semantic_searcher):
    """ingest_pdf 应轮询直到状态变为 completed"""
    from pageindex_rag.ingestion.ingest import DocumentIngestion

    mock_pi_client = MagicMock()
    mock_pi_client.submit_document.return_value = {"doc_id": "cloud-doc-123"}
    # 模拟先 processing 后 completed
    mock_pi_client.get_tree.side_effect = [
        {"status": "processing"},
        {"status": "processing"},
        {"status": "completed", "result": SAMPLE_TREE_SDK_FORMAT}
    ]
    mock_pi_client.delete_document.return_value = None

    with patch("pageindex_rag.ingestion.ingest.PageIndexClient") as mock_client_class, \
         patch("pageindex_rag.ingestion.ingest.generate_doc_description") as mock_gen_desc:

        mock_client_class.return_value = mock_pi_client
        mock_gen_desc.return_value = "测试描述"

        ingestion = DocumentIngestion(
            document_store=mock_document_store,
            semantic_searcher=mock_semantic_searcher,
            config=mock_config
        )

        doc_id = await ingestion.ingest_pdf("/path/to/test.pdf")

        assert doc_id == "pi-test-550e8400"
        assert mock_pi_client.get_tree.call_count == 3


@pytest.mark.asyncio
async def test_ingest_pdf_deletes_cloud_doc(mock_config, mock_document_store, mock_semantic_searcher, mock_pi_client):
    """ingest_pdf 完成后应删除云端文档"""
    from pageindex_rag.ingestion.ingest import DocumentIngestion

    with patch("pageindex_rag.ingestion.ingest.PageIndexClient") as mock_client_class, \
         patch("pageindex_rag.ingestion.ingest.generate_doc_description") as mock_gen_desc:

        mock_client_class.return_value = mock_pi_client
        mock_gen_desc.return_value = "测试描述"

        ingestion = DocumentIngestion(
            document_store=mock_document_store,
            semantic_searcher=mock_semantic_searcher,
            config=mock_config
        )

        await ingestion.ingest_pdf("/path/to/test.pdf")

        mock_pi_client.delete_document.assert_called_once_with("cloud-doc-123")


@pytest.mark.asyncio
async def test_ingest_md_uses_local_md_to_tree(mock_config, mock_document_store, mock_semantic_searcher):
    """ingest_md 应使用本地 md_to_tree 处理 Markdown 文件"""
    from pageindex_rag.ingestion.ingest import DocumentIngestion

    with patch("pageindex_core.page_index_md.md_to_tree", new_callable=AsyncMock) as mock_md_tree, \
         patch("pageindex_rag.ingestion.ingest.generate_doc_description") as mock_gen_desc:

        mock_md_tree.return_value = SAMPLE_MD_TREE
        mock_gen_desc.return_value = "这是一份测试Markdown文档的描述。"

        ingestion = DocumentIngestion(
            document_store=mock_document_store,
            semantic_searcher=mock_semantic_searcher,
            config=mock_config
        )

        doc_id = await ingestion.ingest_md("/path/to/test.md")

        assert doc_id == "pi-test-550e8400"
        mock_md_tree.assert_called_once_with("/path/to/test.md")
        mock_gen_desc.assert_called_once()
        mock_document_store.create.assert_called_once_with(
            "/path/to/test.md",
            SAMPLE_MD_TREE,
            {"doc_description": "这是一份测试Markdown文档的描述。"}
        )
        mock_semantic_searcher.index_document.assert_called_once_with(
            "pi-test-550e8400", SAMPLE_MD_TREE
        )


@pytest.mark.asyncio
async def test_ingest_with_metadata(mock_config, mock_document_store, mock_semantic_searcher, mock_pi_client):
    """测试提供完整 metadata 时不调用 generate_doc_description"""
    from pageindex_rag.ingestion.ingest import DocumentIngestion

    full_metadata = {
        "doc_description": "已提供的文档描述",
        "company": "Test Corp",
        "fiscal_year": "2023",
        "filing_type": "10-K"
    }

    with patch("pageindex_rag.ingestion.ingest.PageIndexClient") as mock_client_class, \
         patch("pageindex_rag.ingestion.ingest.generate_doc_description") as mock_gen_desc:

        mock_client_class.return_value = mock_pi_client

        ingestion = DocumentIngestion(
            document_store=mock_document_store,
            semantic_searcher=mock_semantic_searcher,
            config=mock_config
        )

        doc_id = await ingestion.ingest_pdf("/path/to/test.pdf", metadata=full_metadata)

        assert doc_id == "pi-test-550e8400"
        mock_gen_desc.assert_not_called()
        mock_document_store.create.assert_called_once_with(
            "/path/to/test.pdf",
            SAMPLE_TREE,
            full_metadata
        )
