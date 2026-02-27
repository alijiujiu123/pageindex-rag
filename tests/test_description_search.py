"""Tests for DescriptionSearcher (Issue #6)."""

import json
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from pageindex_rag.search.description_search import DescriptionSearcher


@pytest.fixture
def config():
    return SimpleNamespace(model="gpt-4o-2024-11-20", openai_api_key="test-key")


@pytest.fixture
def sample_docs():
    return [
        {
            "doc_id": "pi-001",
            "doc_name": "apple_10k_2023.pdf",
            "doc_description": "Apple Inc. annual report for fiscal year 2023",
            "company": "Apple",
            "fiscal_year": "2023",
            "filing_type": "10-K",
        },
        {
            "doc_id": "pi-002",
            "doc_name": "microsoft_10q_2023.pdf",
            "doc_description": "Microsoft quarterly report for Q3 2023",
            "company": "Microsoft",
            "fiscal_year": "2023",
            "filing_type": "10-Q",
        },
        {
            "doc_id": "pi-003",
            "doc_name": "tesla_10k_2022.pdf",
            "doc_description": "Tesla annual report for fiscal year 2022",
            "company": "Tesla",
            "fiscal_year": "2022",
            "filing_type": "10-K",
        },
    ]


@pytest.fixture
def document_store(sample_docs):
    store = MagicMock()
    store.query_by_metadata.return_value = sample_docs
    return store


@pytest.mark.asyncio
async def test_description_search(document_store, config, sample_docs):
    """Basic match: LLM returns one doc_id, verify it is returned."""
    llm_response = json.dumps({
        "thinking": "The query is about Apple financials, doc pi-001 matches.",
        "answer": ["pi-001"]
    })

    with patch(
        "pageindex_rag.search.description_search.llm_call_async",
        new=AsyncMock(return_value=llm_response),
    ), patch(
        "pageindex_rag.search.description_search.extract_json",
        return_value={"thinking": "...", "answer": ["pi-001"]},
    ):
        searcher = DescriptionSearcher(document_store, config)
        result = await searcher.search("What is Apple's revenue in 2023?")

    assert result == ["pi-001"]
    document_store.query_by_metadata.assert_called_once()


@pytest.mark.asyncio
async def test_no_match(document_store, config):
    """No match: LLM returns empty answer, result should be empty list."""
    llm_response = json.dumps({
        "thinking": "No document matches this query.",
        "answer": []
    })

    with patch(
        "pageindex_rag.search.description_search.llm_call_async",
        new=AsyncMock(return_value=llm_response),
    ), patch(
        "pageindex_rag.search.description_search.extract_json",
        return_value={"thinking": "No match", "answer": []},
    ):
        searcher = DescriptionSearcher(document_store, config)
        result = await searcher.search("What is the weather today?")

    assert result == []


@pytest.mark.asyncio
async def test_multiple_match(document_store, config):
    """Multiple matches: LLM returns multiple doc_ids, all should be returned."""
    llm_response = json.dumps({
        "thinking": "Both Apple and Microsoft reports are relevant.",
        "answer": ["pi-001", "pi-002"]
    })

    with patch(
        "pageindex_rag.search.description_search.llm_call_async",
        new=AsyncMock(return_value=llm_response),
    ), patch(
        "pageindex_rag.search.description_search.extract_json",
        return_value={"thinking": "...", "answer": ["pi-001", "pi-002"]},
    ):
        searcher = DescriptionSearcher(document_store, config)
        result = await searcher.search("Compare revenue of Apple and Microsoft in 2023?")

    assert result == ["pi-001", "pi-002"]
