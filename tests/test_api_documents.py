"""Tests for FastAPI document management endpoints."""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from pageindex_rag.api.app import app
from pageindex_rag.api.routes.documents import get_document_store, get_ingestion

SAMPLE_DOC_ID = "pi-12345678-1234-1234-1234-123456789abc"

SAMPLE_DOC = {
    "doc_id": SAMPLE_DOC_ID,
    "doc_name": "report.pdf",
    "pdf_path": "/tmp/report.pdf",
    "tree": {"title": "Report", "node_id": "0000", "start_index": 1, "end_index": 10, "nodes": []},
    "created_at": "2024-01-01T00:00:00",
    "doc_description": "A test report",
    "company": "TestCo",
    "fiscal_year": "2023",
    "filing_type": "10-K",
}

SAMPLE_LIST_ITEM = {
    "doc_id": SAMPLE_DOC_ID,
    "doc_name": "report.pdf",
    "created_at": "2024-01-01T00:00:00",
    "company": "TestCo",
    "fiscal_year": "2023",
    "filing_type": "10-K",
}


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.get.return_value = SAMPLE_DOC
    store.list.return_value = [SAMPLE_LIST_ITEM]
    store.query_by_metadata.return_value = [SAMPLE_LIST_ITEM]
    store.delete.return_value = True
    store.update_metadata.return_value = True
    return store


@pytest.fixture
def mock_ingestion():
    ingestion = MagicMock()
    ingestion.ingest_pdf = AsyncMock(return_value=SAMPLE_DOC_ID)
    ingestion.ingest_md = AsyncMock(return_value=SAMPLE_DOC_ID)
    return ingestion


@pytest.fixture
def client(mock_store, mock_ingestion):
    app.dependency_overrides[get_document_store] = lambda: mock_store
    app.dependency_overrides[get_ingestion] = lambda: mock_ingestion
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------- POST /documents ----------

class TestUploadDocument:
    def test_upload_pdf_returns_201_with_doc_id(self, client):
        response = client.post(
            "/documents",
            files={"file": ("report.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
        )
        assert response.status_code == 201
        assert response.json()["doc_id"] == SAMPLE_DOC_ID

    def test_upload_md_returns_201_with_doc_id(self, client, mock_ingestion):
        response = client.post(
            "/documents",
            files={"file": ("doc.md", io.BytesIO(b"# Title"), "text/markdown")},
        )
        assert response.status_code == 201
        assert response.json()["doc_id"] == SAMPLE_DOC_ID
        mock_ingestion.ingest_md.assert_called_once()

    def test_upload_unsupported_format_returns_400(self, client):
        response = client.post(
            "/documents",
            files={"file": ("doc.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert response.status_code == 400

    def test_upload_pdf_calls_ingest_pdf(self, client, mock_ingestion):
        client.post(
            "/documents",
            files={"file": ("report.pdf", io.BytesIO(b"%PDF"), "application/pdf")},
        )
        mock_ingestion.ingest_pdf.assert_called_once()

    def test_upload_with_metadata_params(self, client, mock_ingestion):
        response = client.post(
            "/documents",
            params={"company": "ACME", "fiscal_year": "2023", "filing_type": "10-K"},
            files={"file": ("report.pdf", io.BytesIO(b"%PDF"), "application/pdf")},
        )
        assert response.status_code == 201
        # verify metadata was passed as second positional arg to ingest_pdf
        call_kwargs = mock_ingestion.ingest_pdf.call_args
        metadata = call_kwargs.args[1] if len(call_kwargs.args) > 1 else call_kwargs.kwargs.get("metadata", {})
        assert metadata.get("company") == "ACME"


# ---------- GET /documents ----------

class TestListDocuments:
    def test_list_returns_200(self, client):
        response = client.get("/documents")
        assert response.status_code == 200

    def test_list_returns_list(self, client):
        response = client.get("/documents")
        assert isinstance(response.json(), list)

    def test_list_no_filters_calls_store_list(self, client, mock_store):
        client.get("/documents")
        mock_store.list.assert_called_once()

    def test_list_with_filter_calls_query_by_metadata(self, client, mock_store):
        client.get("/documents?company=TestCo")
        mock_store.query_by_metadata.assert_called_once_with(company="TestCo")

    def test_list_with_multiple_filters(self, client, mock_store):
        client.get("/documents?company=TestCo&filing_type=10-K")
        mock_store.query_by_metadata.assert_called_once_with(company="TestCo", filing_type="10-K")


# ---------- GET /documents/{doc_id} ----------

class TestGetDocument:
    def test_get_existing_returns_200(self, client):
        response = client.get(f"/documents/{SAMPLE_DOC_ID}")
        assert response.status_code == 200

    def test_get_existing_returns_doc(self, client):
        response = client.get(f"/documents/{SAMPLE_DOC_ID}")
        assert response.json()["doc_id"] == SAMPLE_DOC_ID

    def test_get_nonexistent_returns_404(self, client, mock_store):
        mock_store.get.return_value = None
        response = client.get("/documents/pi-nonexistent")
        assert response.status_code == 404

    def test_get_calls_store_get_with_doc_id(self, client, mock_store):
        client.get(f"/documents/{SAMPLE_DOC_ID}")
        mock_store.get.assert_called_once_with(SAMPLE_DOC_ID)


# ---------- DELETE /documents/{doc_id} ----------

class TestDeleteDocument:
    def test_delete_existing_returns_204(self, client):
        response = client.delete(f"/documents/{SAMPLE_DOC_ID}")
        assert response.status_code == 204

    def test_delete_nonexistent_returns_404(self, client, mock_store):
        mock_store.delete.return_value = False
        response = client.delete("/documents/pi-nonexistent")
        assert response.status_code == 404

    def test_delete_calls_store_delete(self, client, mock_store):
        client.delete(f"/documents/{SAMPLE_DOC_ID}")
        mock_store.delete.assert_called_once_with(SAMPLE_DOC_ID)


# ---------- PATCH /documents/{doc_id}/metadata ----------

class TestUpdateMetadata:
    def test_patch_returns_200(self, client):
        response = client.patch(
            f"/documents/{SAMPLE_DOC_ID}/metadata",
            json={"company": "NewCo"},
        )
        assert response.status_code == 200

    def test_patch_returns_updated_true(self, client):
        response = client.patch(
            f"/documents/{SAMPLE_DOC_ID}/metadata",
            json={"company": "NewCo"},
        )
        assert response.json()["updated"] is True

    def test_patch_nonexistent_returns_404(self, client, mock_store):
        mock_store.update_metadata.return_value = False
        response = client.patch(
            "/documents/pi-nonexistent/metadata",
            json={"company": "NewCo"},
        )
        assert response.status_code == 404

    def test_patch_calls_update_metadata(self, client, mock_store):
        client.patch(
            f"/documents/{SAMPLE_DOC_ID}/metadata",
            json={"company": "NewCo", "fiscal_year": "2024"},
        )
        mock_store.update_metadata.assert_called_once_with(
            SAMPLE_DOC_ID, company="NewCo", fiscal_year="2024"
        )
