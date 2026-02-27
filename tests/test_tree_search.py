"""Tests for TreeSearcher (Issue #5)."""

import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from types import SimpleNamespace

from pageindex_rag.retrieval.tree_search import TreeSearcher


SAMPLE_TREE = {
    "title": "Annual Report 2023",
    "node_id": "0000",
    "start_index": 1,
    "end_index": 50,
    "nodes": [
        {
            "title": "Financial Highlights",
            "node_id": "0001",
            "start_index": 1,
            "end_index": 10,
            "nodes": []
        },
        {
            "title": "Risk Factors",
            "node_id": "0002",
            "start_index": 11,
            "end_index": 30,
            "nodes": []
        }
    ]
}


@pytest.fixture
def config():
    return SimpleNamespace(
        model="gpt-4o-2024-11-20",
        openai_api_key="test-api-key",
        openai_base_url="https://api.openai.com/v1",
    )


@pytest.fixture
def searcher(config):
    return TreeSearcher(config)


@pytest.mark.asyncio
async def test_tree_search_basic(searcher):
    """基本搜索，验证返回结构 {thinking, node_list}"""
    mock_response = json.dumps({
        "thinking": "The query is about financial highlights, so node 0001 is relevant.",
        "node_list": ["0001"]
    })

    with patch("pageindex_rag.retrieval.tree_search.llm_call_async", new=AsyncMock(return_value=mock_response)):
        result = await searcher.search(
            query="What are the financial highlights?",
            tree_structure=SAMPLE_TREE
        )

    assert "thinking" in result
    assert "node_list" in result
    assert isinstance(result["thinking"], str)
    assert isinstance(result["node_list"], list)
    assert result["node_list"] == ["0001"]


@pytest.mark.asyncio
async def test_with_preference(searcher):
    """携带 preference/expert_knowledge，验证 Prompt 中包含这些内容"""
    mock_response = json.dumps({
        "thinking": "Using expert knowledge to find relevant nodes.",
        "node_list": ["0001", "0002"]
    })

    expert_knowledge = "Focus on revenue and profit sections"
    preference = "Look for quantitative data"

    with patch("pageindex_rag.retrieval.tree_search.llm_call_async", new=AsyncMock(return_value=mock_response)) as mock_llm:
        result = await searcher.search(
            query="What is the revenue?",
            tree_structure=SAMPLE_TREE,
            expert_knowledge=expert_knowledge,
            preference=preference
        )

    # 验证 prompt 中包含 expert_knowledge 和 preference
    call_args = mock_llm.call_args
    prompt = call_args[0][1]  # positional arg index 1 is the prompt
    assert expert_knowledge in prompt
    assert preference in prompt

    assert result["node_list"] == ["0001", "0002"]


@pytest.mark.asyncio
async def test_json_parse_error(searcher):
    """LLM 返回无效 JSON，返回 {"thinking": "", "node_list": []} 不抛异常"""
    invalid_response = "This is not valid JSON at all!!!"

    with patch("pageindex_rag.retrieval.tree_search.llm_call_async", new=AsyncMock(return_value=invalid_response)):
        with patch("pageindex_rag.retrieval.tree_search.extract_json", side_effect=Exception("JSON parse failed")):
            result = await searcher.search(
                query="What is the revenue?",
                tree_structure=SAMPLE_TREE
            )

    assert result == {"thinking": "", "node_list": []}


@pytest.mark.asyncio
async def test_empty_result(searcher):
    """LLM 返回 node_list=[]，验证返回 {"thinking": "...", "node_list": []}"""
    mock_response = json.dumps({
        "thinking": "No relevant nodes found for this query.",
        "node_list": []
    })

    with patch("pageindex_rag.retrieval.tree_search.llm_call_async", new=AsyncMock(return_value=mock_response)):
        result = await searcher.search(
            query="What is the weather today?",
            tree_structure=SAMPLE_TREE
        )

    assert result["thinking"] == "No relevant nodes found for this query."
    assert result["node_list"] == []
