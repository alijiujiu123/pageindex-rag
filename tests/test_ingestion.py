import pytest
from unittest.mock import AsyncMock, MagicMock, patch


SAMPLE_TREE = {
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

SAMPLE_MD_CONTENT = "# Test Document\n\n## Chapter 1\n\nSome content here.\n"


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


@pytest.mark.asyncio
async def test_ingest_pdf_uses_markitdown_not_page_index_main(mock_document_store, mock_semantic_searcher):
    """ingest_pdf 应使用 markitdown 转换，不再调用 page_index_main"""
    from pageindex_rag.ingestion.ingest import DocumentIngestion

    mock_markitdown_result = MagicMock()
    mock_markitdown_result.text_content = SAMPLE_MD_CONTENT

    with patch("pageindex_rag.ingestion.ingest.MarkItDown") as mock_md_class, \
         patch("pageindex_rag.ingestion.ingest.md_to_tree", new_callable=AsyncMock) as mock_md_tree, \
         patch("pageindex_rag.ingestion.ingest.generate_doc_description") as mock_gen_desc, \
         patch("pageindex_rag.ingestion.ingest.page_index_main") as mock_page_index:

        mock_md_instance = MagicMock()
        mock_md_instance.convert.return_value = mock_markitdown_result
        mock_md_class.return_value = mock_md_instance
        mock_md_tree.return_value = SAMPLE_TREE
        mock_gen_desc.return_value = "这是一份测试PDF文档的描述。"

        ingestion = DocumentIngestion(
            document_store=mock_document_store,
            semantic_searcher=mock_semantic_searcher
        )

        doc_id = await ingestion.ingest_pdf("/path/to/test.pdf")

        # page_index_main 不应被调用
        mock_page_index.assert_not_called()
        # MarkItDown 应被调用并转换 PDF
        mock_md_instance.convert.assert_called_once_with("/path/to/test.pdf")
        # md_to_tree 应被调用（传入临时文件路径）
        mock_md_tree.assert_called_once()
        assert doc_id == "pi-test-550e8400"


@pytest.mark.asyncio
async def test_ingest_pdf_generates_description(mock_document_store, mock_semantic_searcher):
    """ingest_pdf 无 doc_description 时应调用 generate_doc_description"""
    from pageindex_rag.ingestion.ingest import DocumentIngestion

    mock_markitdown_result = MagicMock()
    mock_markitdown_result.text_content = SAMPLE_MD_CONTENT

    with patch("pageindex_rag.ingestion.ingest.MarkItDown") as mock_md_class, \
         patch("pageindex_rag.ingestion.ingest.md_to_tree", new_callable=AsyncMock) as mock_md_tree, \
         patch("pageindex_rag.ingestion.ingest.generate_doc_description") as mock_gen_desc:

        mock_md_instance = MagicMock()
        mock_md_instance.convert.return_value = mock_markitdown_result
        mock_md_class.return_value = mock_md_instance
        mock_md_tree.return_value = SAMPLE_TREE
        mock_gen_desc.return_value = "这是一份测试PDF文档的描述。"

        ingestion = DocumentIngestion(
            document_store=mock_document_store,
            semantic_searcher=mock_semantic_searcher
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
async def test_ingest_md(mock_document_store, mock_semantic_searcher):
    """测试 Markdown 文档入库流程"""
    from pageindex_rag.ingestion.ingest import DocumentIngestion

    with patch("pageindex_rag.ingestion.ingest.md_to_tree", new_callable=AsyncMock) as mock_md_tree, \
         patch("pageindex_rag.ingestion.ingest.generate_doc_description") as mock_gen_desc:

        mock_md_tree.return_value = SAMPLE_TREE
        mock_gen_desc.return_value = "这是一份测试Markdown文档的描述。"

        ingestion = DocumentIngestion(
            document_store=mock_document_store,
            semantic_searcher=mock_semantic_searcher
        )

        doc_id = await ingestion.ingest_md("/path/to/test.md")

        assert doc_id == "pi-test-550e8400"
        mock_md_tree.assert_called_once_with("/path/to/test.md")
        mock_gen_desc.assert_called_once()
        mock_document_store.create.assert_called_once_with(
            "/path/to/test.md",
            SAMPLE_TREE,
            {"doc_description": "这是一份测试Markdown文档的描述。"}
        )
        mock_semantic_searcher.index_document.assert_called_once_with(
            "pi-test-550e8400", SAMPLE_TREE
        )


@pytest.mark.asyncio
async def test_ingest_with_metadata(mock_document_store, mock_semantic_searcher):
    """测试提供完整 metadata 时不调用 generate_doc_description"""
    from pageindex_rag.ingestion.ingest import DocumentIngestion

    full_metadata = {
        "doc_description": "已提供的文档描述",
        "company": "Test Corp",
        "fiscal_year": "2023",
        "filing_type": "10-K"
    }

    mock_markitdown_result = MagicMock()
    mock_markitdown_result.text_content = SAMPLE_MD_CONTENT

    with patch("pageindex_rag.ingestion.ingest.MarkItDown") as mock_md_class, \
         patch("pageindex_rag.ingestion.ingest.md_to_tree", new_callable=AsyncMock) as mock_md_tree, \
         patch("pageindex_rag.ingestion.ingest.generate_doc_description") as mock_gen_desc:

        mock_md_instance = MagicMock()
        mock_md_instance.convert.return_value = mock_markitdown_result
        mock_md_class.return_value = mock_md_instance
        mock_md_tree.return_value = SAMPLE_TREE

        ingestion = DocumentIngestion(
            document_store=mock_document_store,
            semantic_searcher=mock_semantic_searcher
        )

        doc_id = await ingestion.ingest_pdf("/path/to/test.pdf", metadata=full_metadata)

        assert doc_id == "pi-test-550e8400"
        mock_gen_desc.assert_not_called()
        mock_document_store.create.assert_called_once_with(
            "/path/to/test.pdf",
            SAMPLE_TREE,
            full_metadata
        )
