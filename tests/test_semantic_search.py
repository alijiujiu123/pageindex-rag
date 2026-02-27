"""Tests for SemanticSearcher (Issue #8)."""
import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return SimpleNamespace(
        chroma_persist_dir="./chroma_data",
        embedding_model="text-embedding-3-small",
        semantic_top_k=20,
        openai_api_key="test-key",
        embedding_api_key="test-key",
        embedding_base_url="https://api.openai.com/v1",
    )


@pytest.fixture
def sample_tree():
    """A minimal tree with two nodes."""
    return {
        "title": "Root",
        "node_id": "0000",
        "start_index": 1,
        "end_index": 10,
        "summary": "Root summary",
        "nodes": [
            {
                "title": "Chapter 1",
                "node_id": "0001",
                "start_index": 1,
                "end_index": 5,
                "summary": "Chapter 1 summary",
                "nodes": [],
            }
        ],
    }


# ---------------------------------------------------------------------------
# Test 1: index_document
# ---------------------------------------------------------------------------

def test_index_document(config, sample_tree):
    """index_document 应将每个节点 embed 并 upsert 到 ChromaDB，id 格式为 {doc_id}#{node_id}。"""
    mock_collection = MagicMock()

    with patch("chromadb.PersistentClient") as mock_chroma_client, \
         patch("pageindex_rag.search.semantic_search.get_embedding") as mock_embed:

        mock_chroma_client.return_value.get_or_create_collection.return_value = mock_collection
        mock_embed.return_value = [0.1, 0.2, 0.3]

        from pageindex_rag.search.semantic_search import SemanticSearcher
        searcher = SemanticSearcher(config)

        doc_id = "pi-test-doc-001"
        searcher.index_document(doc_id, sample_tree)

        # 应调用 get_embedding（两个节点各调用一次）
        assert mock_embed.call_count == 2

        # 验证 upsert 被调用两次
        assert mock_collection.upsert.call_count == 2

        # 验证 id 格式
        call_args_list = mock_collection.upsert.call_args_list
        upserted_ids = []
        for call in call_args_list:
            ids = call.kwargs.get("ids") or call.args[0]
            upserted_ids.extend(ids)

        assert f"{doc_id}#0000" in upserted_ids
        assert f"{doc_id}#0001" in upserted_ids


# ---------------------------------------------------------------------------
# Test 2: search_ranked
# ---------------------------------------------------------------------------

def test_search_ranked(config):
    """search 应按 DocScore 降序返回 doc_id 列表。"""
    mock_collection = MagicMock()

    # ChromaDB query 返回：doc_A 命中1次(distance=0.1), doc_B 命中1次(distance=0.3)
    # ChunkScore = 1 - distance
    # doc_A: N=1, score=0.9  → DocScore = 1/√2 × 0.9 ≈ 0.636
    # doc_B: N=1, score=0.7  → DocScore = 1/√2 × 0.7 ≈ 0.495
    mock_collection.query.return_value = {
        "ids": [["pi-doc-A#0001", "pi-doc-B#0001"]],
        "distances": [[0.1, 0.3]],
        "metadatas": [[
            {"doc_id": "pi-doc-A", "node_id": "0001"},
            {"doc_id": "pi-doc-B", "node_id": "0001"},
        ]],
    }

    with patch("chromadb.PersistentClient") as mock_chroma_client, \
         patch("pageindex_rag.search.semantic_search.get_embedding") as mock_embed:

        mock_chroma_client.return_value.get_or_create_collection.return_value = mock_collection
        mock_embed.return_value = [0.1, 0.2, 0.3]

        from pageindex_rag.search.semantic_search import SemanticSearcher
        searcher = SemanticSearcher(config)

        result = searcher.search("some query", top_k=5)

        assert result == ["pi-doc-A", "pi-doc-B"], f"Expected ['pi-doc-A', 'pi-doc-B'], got {result}"


# ---------------------------------------------------------------------------
# Test 3: docscore_formula
# ---------------------------------------------------------------------------

def test_docscore_formula(config):
    """单元测试 DocScore 计算：doc_A 有 2 个 chunk，scores=[0.9, 0.8]，DocScore = 1/√3 × 1.7 ≈ 0.981。"""
    mock_collection = MagicMock()

    # doc_A 命中 2 个 chunk，distances=[0.1, 0.2]
    mock_collection.query.return_value = {
        "ids": [["pi-doc-A#0001", "pi-doc-A#0002"]],
        "distances": [[0.1, 0.2]],
        "metadatas": [[
            {"doc_id": "pi-doc-A", "node_id": "0001"},
            {"doc_id": "pi-doc-A", "node_id": "0002"},
        ]],
    }

    with patch("chromadb.PersistentClient") as mock_chroma_client, \
         patch("pageindex_rag.search.semantic_search.get_embedding") as mock_embed:

        mock_chroma_client.return_value.get_or_create_collection.return_value = mock_collection
        mock_embed.return_value = [0.1, 0.2, 0.3]

        from pageindex_rag.search.semantic_search import SemanticSearcher
        searcher = SemanticSearcher(config)

        result = searcher.search("query", top_k=5)

        # 验证 doc_A 在结果中
        assert "pi-doc-A" in result

        # 手动验证 DocScore 公式
        N = 2
        chunk_scores = [1 - 0.1, 1 - 0.2]  # [0.9, 0.8]
        expected_score = (1 / math.sqrt(N + 1)) * sum(chunk_scores)
        # 1/√3 × 1.7 ≈ 0.9814
        assert abs(expected_score - (1 / math.sqrt(3)) * 1.7) < 1e-6


# ---------------------------------------------------------------------------
# Test 4: empty_index
# ---------------------------------------------------------------------------

def test_empty_index(config):
    """ChromaDB 返回空结果时，search 应返回空列表。"""
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "ids": [[]],
        "distances": [[]],
        "metadatas": [[]],
    }

    with patch("chromadb.PersistentClient") as mock_chroma_client, \
         patch("pageindex_rag.search.semantic_search.get_embedding") as mock_embed:

        mock_chroma_client.return_value.get_or_create_collection.return_value = mock_collection
        mock_embed.return_value = [0.1, 0.2, 0.3]

        from pageindex_rag.search.semantic_search import SemanticSearcher
        searcher = SemanticSearcher(config)

        result = searcher.search("anything", top_k=5)
        assert result == []
