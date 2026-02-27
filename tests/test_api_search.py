"""Tests for FastAPI search and QA endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from pageindex_rag.api.app import app
from pageindex_rag.api.routes.search import get_search_router
from pageindex_rag.api.routes.documents import get_document_store, get_tree_searcher
from pageindex_rag.api.routes.qa import get_rag_pipeline

SAMPLE_DOC_ID = "pi-12345678-1234-1234-1234-123456789abc"
SAMPLE_DOC_ID_2 = "pi-87654321-4321-4321-4321-cba987654321"

SAMPLE_DOC = {
    "doc_id": SAMPLE_DOC_ID,
    "doc_name": "report.pdf",
    "pdf_path": "/tmp/report.pdf",
    "tree": {
        "title": "Report",
        "node_id": "0000",
        "start_index": 1,
        "end_index": 10,
        "nodes": [
            {
                "title": "Chapter 1",
                "node_id": "0001",
                "start_index": 1,
                "end_index": 5,
                "summary": "Introduction",
                "nodes": [],
            },
            {
                "title": "Chapter 2",
                "node_id": "0002",
                "start_index": 6,
                "end_index": 10,
                "summary": "Analysis",
                "nodes": [],
            },
        ],
    },
    "created_at": "2024-01-01T00:00:00",
    "doc_description": "A test report",
    "company": "TestCo",
    "fiscal_year": "2023",
    "filing_type": "10-K",
}


@pytest.fixture
def mock_search_router():
    router = MagicMock()
    router.search = AsyncMock(return_value=[SAMPLE_DOC_ID, SAMPLE_DOC_ID_2])
    return router


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.get.return_value = SAMPLE_DOC
    return store


@pytest.fixture
def mock_tree_searcher():
    searcher = MagicMock()
    searcher.search = AsyncMock(
        return_value={
            "thinking": "Search reasoning",
            "node_list": ["0001", "0002"],
        }
    )
    return searcher


@pytest.fixture
def mock_rag_pipeline():
    pipeline = MagicMock()
    pipeline.query = AsyncMock(
        return_value={
            "answer": "This is a test answer.",
            "sources": [
                {
                    "doc_id": SAMPLE_DOC_ID,
                    "node_id": "0001",
                    "page_range": "1-5",
                }
            ],
        }
    )
    return pipeline


@pytest.fixture
def client_search(mock_search_router):
    app.dependency_overrides[get_search_router] = lambda: mock_search_router
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def client_documents(mock_store, mock_tree_searcher):
    app.dependency_overrides[get_document_store] = lambda: mock_store
    app.dependency_overrides[get_tree_searcher] = lambda: mock_tree_searcher
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def client_qa(mock_rag_pipeline):
    app.dependency_overrides[get_rag_pipeline] = lambda: mock_rag_pipeline
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------- POST /search ----------

class TestSearchEndpoint:
    def test_search_returns_200(self, client_search):
        response = client_search.post("/search", json={"query": "test question"})
        assert response.status_code == 200

    def test_search_returns_results(self, client_search):
        response = client_search.post("/search", json={"query": "test question"})
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_search_returns_doc_ids(self, client_search):
        response = client_search.post("/search", json={"query": "test question"})
        data = response.json()
        doc_ids = [r["doc_id"] for r in data["results"]]
        assert SAMPLE_DOC_ID in doc_ids
        assert SAMPLE_DOC_ID_2 in doc_ids

    def test_search_calls_search_router(self, client_search, mock_search_router):
        client_search.post("/search", json={"query": "test question"})
        mock_search_router.search.assert_called_once_with("test question")


# ---------- POST /documents/{doc_id}/search ----------

class TestTreeSearchEndpoint:
    def test_tree_search_returns_200(self, client_documents):
        response = client_documents.post(
            f"/documents/{SAMPLE_DOC_ID}/search",
            json={"query": "test question"},
        )
        assert response.status_code == 200

    def test_tree_search_returns_doc_id(self, client_documents):
        response = client_documents.post(
            f"/documents/{SAMPLE_DOC_ID}/search",
            json={"query": "test question"},
        )
        data = response.json()
        assert data["doc_id"] == SAMPLE_DOC_ID

    def test_tree_search_returns_nodes(self, client_documents):
        response = client_documents.post(
            f"/documents/{SAMPLE_DOC_ID}/search",
            json={"query": "test question"},
        )
        data = response.json()
        assert "nodes" in data
        assert isinstance(data["nodes"], list)

    def test_tree_search_returns_node_details(self, client_documents):
        response = client_documents.post(
            f"/documents/{SAMPLE_DOC_ID}/search",
            json={"query": "test question"},
        )
        data = response.json()
        nodes = data["nodes"]
        assert len(nodes) == 2
        assert nodes[0]["node_id"] == "0001"
        assert nodes[0]["title"] == "Chapter 1"
        assert nodes[0]["summary"] == "Introduction"
        assert nodes[1]["node_id"] == "0002"
        assert nodes[1]["title"] == "Chapter 2"

    def test_tree_search_nonexistent_doc_returns_404(self, client_documents, mock_store):
        mock_store.get.return_value = None
        response = client_documents.post(
            "/documents/pi-nonexistent/search",
            json={"query": "test question"},
        )
        assert response.status_code == 404

    def test_tree_search_calls_tree_searcher(self, client_documents, mock_tree_searcher):
        client_documents.post(
            f"/documents/{SAMPLE_DOC_ID}/search",
            json={"query": "test question"},
        )
        mock_tree_searcher.search.assert_called_once()
        call_args = mock_tree_searcher.search.call_args
        assert call_args.args[0] == "test question"


# ---------- POST /qa ----------

class TestQAEndpointMultiDoc:
    def test_qa_multi_doc_returns_200(self, client_qa):
        response = client_qa.post(
            "/qa",
            json={"query": "test question"},
        )
        assert response.status_code == 200

    def test_qa_multi_doc_returns_answer(self, client_qa):
        response = client_qa.post(
            "/qa",
            json={"query": "test question"},
        )
        data = response.json()
        assert "answer" in data
        assert data["answer"] == "This is a test answer."

    def test_qa_multi_doc_returns_sources(self, client_qa):
        response = client_qa.post(
            "/qa",
            json={"query": "test question"},
        )
        data = response.json()
        assert "sources" in data
        assert isinstance(data["sources"], list)

    def test_qa_multi_doc_source_structure(self, client_qa):
        response = client_qa.post(
            "/qa",
            json={"query": "test question"},
        )
        data = response.json()
        sources = data["sources"]
        assert len(sources) == 1
        assert sources[0]["doc_id"] == SAMPLE_DOC_ID
        assert sources[0]["node_id"] == "0001"
        assert sources[0]["page_range"] == "1-5"

    def test_qa_multi_doc_calls_pipeline_without_doc_id(self, client_qa, mock_rag_pipeline):
        client_qa.post("/qa", json={"query": "test question"})
        mock_rag_pipeline.query.assert_called_once_with("test question", None)


class TestQAEndpointSingleDoc:
    def test_qa_single_doc_returns_200(self, client_qa):
        response = client_qa.post(
            "/qa",
            json={"query": "test question", "doc_id": SAMPLE_DOC_ID},
        )
        assert response.status_code == 200

    def test_qa_single_doc_returns_answer(self, client_qa):
        response = client_qa.post(
            "/qa",
            json={"query": "test question", "doc_id": SAMPLE_DOC_ID},
        )
        data = response.json()
        assert "answer" in data

    def test_qa_single_doc_calls_pipeline_with_doc_id(self, client_qa, mock_rag_pipeline):
        client_qa.post(
            "/qa",
            json={"query": "test question", "doc_id": SAMPLE_DOC_ID},
        )
        mock_rag_pipeline.query.assert_called_once_with("test question", SAMPLE_DOC_ID)
