import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def mock_document_store():
    store = MagicMock()
    store.get.return_value = {
        "doc_id": "pi-test-123",
        "tree_json": {
            "title": "Test Document",
            "node_id": "0001",
            "start_index": 1,
            "end_index": 10,
            "nodes": []
        }
    }
    return store


@pytest.fixture
def mock_tree_searcher():
    searcher = MagicMock()
    searcher.search = AsyncMock(return_value=["0001", "0002"])
    return searcher


@pytest.fixture
def mock_node_extractor():
    extractor = MagicMock()
    extractor.extract.return_value = [
        {"node_id": "0001", "content": "Test content 1", "page_range": "1-5"},
        {"node_id": "0002", "content": "Test content 2", "page_range": "6-10"}
    ]
    return extractor


@pytest.fixture
def mock_search_router():
    router = MagicMock()
    router.search = AsyncMock(return_value=[
        {"doc_id": "pi-test-123", "score": 0.9},
        {"doc_id": "pi-test-456", "score": 0.8}
    ])
    return router


@pytest.mark.asyncio
async def test_single_doc_pipeline(mock_document_store, mock_tree_searcher, mock_node_extractor):
    """测试单文档模式的 RAG Pipeline"""
    from pageindex_rag.pipeline.rag_pipeline import RAGPipeline

    with patch("pageindex_rag.pipeline.answer_generator.llm_call_async", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = "这是基于文档内容生成的答案。"

        pipeline = RAGPipeline(
            document_store=mock_document_store,
            tree_searcher=mock_tree_searcher,
            node_extractor=mock_node_extractor
        )

        result = await pipeline.query("测试问题", doc_id="pi-test-123")

        assert "answer" in result
        assert "sources" in result
        assert result["answer"] == "这是基于文档内容生成的答案。"
        assert len(result["sources"]) == 2
        assert result["sources"][0]["doc_id"] == "pi-test-123"
        assert result["sources"][0]["node_id"] == "0001"

        mock_document_store.get.assert_called_once_with("pi-test-123")
        mock_tree_searcher.search.assert_called_once()
        mock_node_extractor.extract.assert_called_once_with("pi-test-123", ["0001", "0002"])


@pytest.mark.asyncio
async def test_multi_doc_pipeline(mock_document_store, mock_tree_searcher, mock_node_extractor, mock_search_router):
    """测试多文档模式的 RAG Pipeline"""
    from pageindex_rag.pipeline.rag_pipeline import RAGPipeline

    # 第二个文档
    def get_doc(doc_id):
        docs = {
            "pi-test-123": {
                "doc_id": "pi-test-123",
                "tree_json": {"title": "Doc 1", "node_id": "0001", "start_index": 1, "end_index": 10, "nodes": []}
            },
            "pi-test-456": {
                "doc_id": "pi-test-456",
                "tree_json": {"title": "Doc 2", "node_id": "0001", "start_index": 1, "end_index": 20, "nodes": []}
            }
        }
        return docs.get(doc_id)

    mock_document_store.get.side_effect = get_doc

    with patch("pageindex_rag.pipeline.answer_generator.llm_call_async", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = "基于多个文档生成的综合答案。"

        pipeline = RAGPipeline(
            document_store=mock_document_store,
            tree_searcher=mock_tree_searcher,
            node_extractor=mock_node_extractor,
            search_router=mock_search_router
        )

        result = await pipeline.query("跨文档问题")

        assert "answer" in result
        assert "sources" in result
        assert result["answer"] == "基于多个文档生成的综合答案。"
        assert len(result["sources"]) == 4  # 2个文档，每个2个节点

        mock_search_router.search.assert_called_once_with("跨文档问题")
        assert mock_tree_searcher.search.call_count == 2
        assert mock_node_extractor.extract.call_count == 2


@pytest.mark.asyncio
async def test_no_relevant_doc(mock_document_store, mock_tree_searcher, mock_node_extractor):
    """测试 SearchRouter 返回空列表的情况"""
    from pageindex_rag.pipeline.rag_pipeline import RAGPipeline

    empty_router = MagicMock()
    empty_router.search = AsyncMock(return_value=[])

    pipeline = RAGPipeline(
        document_store=mock_document_store,
        tree_searcher=mock_tree_searcher,
        node_extractor=mock_node_extractor,
        search_router=empty_router
    )

    result = await pipeline.query("无相关文档的问题")

    assert "answer" in result
    assert "sources" in result
    assert result["sources"] == []
    assert "未找到" in result["answer"] or "no" in result["answer"].lower()


@pytest.mark.asyncio
async def test_single_doc_pipeline_handles_tree_search_dict_and_extractor_map():
    """单文档模式兼容 tree_search dict 输出与 extractor map 输出。"""
    from pageindex_rag.pipeline.rag_pipeline import RAGPipeline

    document_store = MagicMock()
    document_store.get.return_value = {
        "doc_id": "pi-test-123",
        "tree": {
            "title": "Doc",
            "node_id": "0000",
            "start_index": 1,
            "end_index": 2,
            "nodes": [
                {"node_id": "0001", "title": "S1", "start_index": 1, "end_index": 1, "nodes": []},
                {"node_id": "0002", "title": "S2", "start_index": 2, "end_index": 2, "nodes": []},
            ],
        },
    }
    tree_searcher = MagicMock()
    tree_searcher.search = AsyncMock(return_value={"thinking": "...", "node_list": ["0001", "0002"]})
    node_extractor = MagicMock()
    node_extractor.extract.return_value = {"0001": "text-1", "0002": "text-2"}

    with patch("pageindex_rag.pipeline.answer_generator.llm_call_async", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = "answer"
        pipeline = RAGPipeline(document_store, tree_searcher, node_extractor)
        result = await pipeline.query("Q", doc_id="pi-test-123")

    assert result["answer"] == "answer"
    assert len(result["sources"]) == 2
    assert result["sources"][0]["node_id"] == "0001"
    assert result["sources"][0]["page_range"] == "1-1"
    assert result["sources"][1]["node_id"] == "0002"
    assert result["sources"][1]["page_range"] == "2-2"


@pytest.mark.asyncio
async def test_multi_doc_pipeline_handles_router_doc_id_list():
    """多文档模式兼容 search_router 返回 list[str]。"""
    from pageindex_rag.pipeline.rag_pipeline import RAGPipeline

    document_store = MagicMock()
    document_store.get.side_effect = lambda doc_id: {
        "doc_id": doc_id,
        "tree": {
            "title": "Doc",
            "node_id": "0000",
            "start_index": 1,
            "end_index": 1,
            "nodes": [{"node_id": "0001", "title": "S1", "start_index": 1, "end_index": 1, "nodes": []}],
        },
    }
    tree_searcher = MagicMock()
    tree_searcher.search = AsyncMock(return_value={"thinking": "...", "node_list": ["0001"]})
    node_extractor = MagicMock()
    node_extractor.extract.return_value = {"0001": "text"}
    search_router = MagicMock()
    search_router.search = AsyncMock(return_value=["pi-test-123", "pi-test-456"])

    with patch("pageindex_rag.pipeline.answer_generator.llm_call_async", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = "answer"
        pipeline = RAGPipeline(
            document_store=document_store,
            tree_searcher=tree_searcher,
            node_extractor=node_extractor,
            search_router=search_router,
        )
        result = await pipeline.query("Q")

    assert result["answer"] == "answer"
    assert [s["doc_id"] for s in result["sources"]] == ["pi-test-123", "pi-test-456"]
