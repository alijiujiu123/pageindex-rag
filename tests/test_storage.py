"""Tests for DocumentStore - document CRUD operations."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from pageindex_rag.storage.models import Base, Document
from pageindex_rag.storage.document_store import DocumentStore


@pytest.fixture
def engine():
    """In-memory SQLite engine for tests."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)


@pytest.fixture
def store(engine):
    """DocumentStore backed by in-memory SQLite."""
    Session = sessionmaker(bind=engine)
    return DocumentStore(Session)


@pytest.fixture
def sample_tree():
    return {
        "title": "Annual Report 2023",
        "node_id": "0000",
        "start_index": 1,
        "end_index": 50,
        "nodes": [
            {"title": "Chapter 1", "node_id": "0001", "start_index": 1, "end_index": 25},
            {"title": "Chapter 2", "node_id": "0002", "start_index": 26, "end_index": 50},
        ],
    }


class TestCreateDocument:
    def test_create_returns_doc_id_with_pi_prefix(self, store, sample_tree):
        doc_id = store.create("report.pdf", sample_tree)
        assert doc_id.startswith("pi-")

    def test_create_doc_id_is_unique(self, store, sample_tree):
        id1 = store.create("report.pdf", sample_tree)
        id2 = store.create("report.pdf", sample_tree)
        assert id1 != id2

    def test_create_stores_tree_json(self, store, sample_tree):
        doc_id = store.create("report.pdf", sample_tree)
        doc = store.get(doc_id)
        assert doc["tree"]["title"] == "Annual Report 2023"
        assert len(doc["tree"]["nodes"]) == 2

    def test_create_stores_pdf_path(self, store, sample_tree):
        doc_id = store.create("/data/report.pdf", sample_tree)
        doc = store.get(doc_id)
        assert doc["pdf_path"] == "/data/report.pdf"

    def test_create_stores_doc_name_from_filename(self, store, sample_tree):
        doc_id = store.create("/data/annual_report.pdf", sample_tree)
        doc = store.get(doc_id)
        assert doc["doc_name"] == "annual_report.pdf"


class TestGetDocument:
    def test_get_returns_none_for_missing_doc_id(self, store):
        result = store.get("pi-nonexistent")
        assert result is None

    def test_get_returns_doc_id(self, store, sample_tree):
        doc_id = store.create("report.pdf", sample_tree)
        doc = store.get(doc_id)
        assert doc["doc_id"] == doc_id

    def test_get_returns_created_at(self, store, sample_tree):
        doc_id = store.create("report.pdf", sample_tree)
        doc = store.get(doc_id)
        assert doc["created_at"] is not None


class TestListDocuments:
    def test_list_returns_empty_when_no_documents(self, store):
        assert store.list() == []

    def test_list_returns_all_documents(self, store, sample_tree):
        store.create("report1.pdf", sample_tree)
        store.create("report2.pdf", sample_tree)
        docs = store.list()
        assert len(docs) == 2

    def test_list_returns_doc_ids_and_names(self, store, sample_tree):
        store.create("report.pdf", sample_tree)
        docs = store.list()
        assert "doc_id" in docs[0]
        assert "doc_name" in docs[0]
        assert "created_at" in docs[0]

    def test_list_does_not_return_tree_json(self, store, sample_tree):
        store.create("report.pdf", sample_tree)
        docs = store.list()
        assert "tree" not in docs[0]


class TestDeleteDocument:
    def test_delete_existing_document_returns_true(self, store, sample_tree):
        doc_id = store.create("report.pdf", sample_tree)
        assert store.delete(doc_id) is True

    def test_delete_removes_document(self, store, sample_tree):
        doc_id = store.create("report.pdf", sample_tree)
        store.delete(doc_id)
        assert store.get(doc_id) is None

    def test_delete_nonexistent_returns_false(self, store):
        assert store.delete("pi-nonexistent") is False

    def test_delete_only_removes_target_document(self, store, sample_tree):
        id1 = store.create("report1.pdf", sample_tree)
        id2 = store.create("report2.pdf", sample_tree)
        store.delete(id1)
        assert store.get(id2) is not None
