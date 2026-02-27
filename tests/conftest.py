"""Shared test fixtures for PageIndex RAG tests."""

import pytest
from pageindex_rag.config import get_config


@pytest.fixture
def config():
    """Return a test configuration (no real DB or API calls)."""
    return get_config(
        database_url="sqlite:///test.db",
        openai_api_key="test-key",
        chroma_persist_dir="/tmp/chroma_test",
    )


@pytest.fixture
def sample_tree():
    """Return a minimal PageIndex tree structure for testing."""
    return {
        "title": "Test Document",
        "node_id": "0000",
        "start_index": 1,
        "end_index": 10,
        "nodes": [
            {
                "title": "Chapter 1",
                "node_id": "0001",
                "start_index": 1,
                "end_index": 5,
            },
            {
                "title": "Chapter 2",
                "node_id": "0002",
                "start_index": 6,
                "end_index": 10,
            },
        ],
    }
