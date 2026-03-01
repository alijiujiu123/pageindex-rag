"""Tests for MetadataSearcher (Issue #7)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace

from pageindex_rag.search.metadata_search import MetadataSearcher


@pytest.fixture
def config():
    return SimpleNamespace(
        model="gpt-4o-2024-11-20",
        openai_api_key="test-key",
    )


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.query_by_metadata = MagicMock(return_value=[])
    return store


@pytest.mark.asyncio
async def test_query_to_sql(mock_store, config):
    """LLM 返回含 null 字段时，null 字段应被过滤，只传有效字段给 query_by_metadata。"""
    llm_response = '{"company": "Apple", "fiscal_year": "2023", "filing_type": null}'

    with patch(
        "pageindex_rag.search.metadata_search.llm_call_async",
        new=AsyncMock(return_value=llm_response),
    ):
        searcher = MetadataSearcher(mock_store, config)
        await searcher.search("Apple 2023 annual report")

    mock_store.query_by_metadata.assert_called_once_with(
        company="Apple", fiscal_year="2023"
    )


@pytest.mark.asyncio
async def test_sql_injection_prevention(mock_store, config):
    """注入字符串应直接作为参数传给 ORM，由 ORM 参数化防注入，MetadataSearcher 层不应过滤。"""
    injection_str = "'; DROP TABLE documents; --"
    llm_response = f'{{"company": "{injection_str}", "fiscal_year": null, "filing_type": null}}'

    with patch(
        "pageindex_rag.search.metadata_search.llm_call_async",
        new=AsyncMock(return_value=llm_response),
    ):
        searcher = MetadataSearcher(mock_store, config)
        await searcher.search("some query")

    mock_store.query_by_metadata.assert_called_once_with(company=injection_str)


@pytest.mark.asyncio
async def test_end_to_end(mock_store, config):
    """完整流程：LLM 提取条件 → store.query_by_metadata → 返回 doc_id 列表。"""
    llm_response = '{"company": "Microsoft", "fiscal_year": "2022", "filing_type": "10-K"}'
    mock_store.query_by_metadata.return_value = [
        {"doc_id": "pi-aaa", "doc_name": "msft_2022.pdf"},
        {"doc_id": "pi-bbb", "doc_name": "msft_2022_q4.pdf"},
    ]

    with patch(
        "pageindex_rag.search.metadata_search.llm_call_async",
        new=AsyncMock(return_value=llm_response),
    ):
        searcher = MetadataSearcher(mock_store, config)
        result = await searcher.search("Microsoft 2022 10-K filing")

    assert result == ["pi-aaa", "pi-bbb"]
    mock_store.query_by_metadata.assert_called_once_with(
        company="Microsoft", fiscal_year="2022", filing_type="10-K"
    )
