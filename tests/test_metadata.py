"""Tests for Issue #3: Document Metadata Management.

Tests cover:
- save_metadata: storing metadata when creating documents
- update_metadata: updating metadata fields for existing documents
- query_by_metadata: SQL-filtered queries by company/fiscal_year/filing_type
- test_metadata_sql_filter: parameterized query protection against injection
- get/list return metadata fields
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from pageindex_rag.storage.models import Base
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
        "title": "Apple 10-K 2023",
        "node_id": "0000",
        "start_index": 1,
        "end_index": 100,
        "nodes": [],
    }


class TestSaveMetadata:
    def test_save_metadata_stores_company(self, store, sample_tree):
        """create() accepts metadata dict with company field."""
        doc_id = store.create(
            "apple_2023.pdf",
            sample_tree,
            metadata={"company": "Apple"},
        )
        doc = store.get(doc_id)
        assert doc["company"] == "Apple"

    def test_save_metadata_stores_fiscal_year(self, store, sample_tree):
        doc_id = store.create(
            "apple_2023.pdf",
            sample_tree,
            metadata={"fiscal_year": "2023"},
        )
        doc = store.get(doc_id)
        assert doc["fiscal_year"] == "2023"

    def test_save_metadata_stores_filing_type(self, store, sample_tree):
        doc_id = store.create(
            "apple_2023.pdf",
            sample_tree,
            metadata={"filing_type": "10-K"},
        )
        doc = store.get(doc_id)
        assert doc["filing_type"] == "10-K"

    def test_save_metadata_stores_doc_description(self, store, sample_tree):
        doc_id = store.create(
            "apple_2023.pdf",
            sample_tree,
            metadata={"doc_description": "Apple annual report 2023"},
        )
        doc = store.get(doc_id)
        assert doc["doc_description"] == "Apple annual report 2023"

    def test_save_metadata_all_fields_together(self, store, sample_tree):
        doc_id = store.create(
            "apple_2023.pdf",
            sample_tree,
            metadata={
                "company": "Apple",
                "fiscal_year": "2023",
                "filing_type": "10-K",
                "doc_description": "Apple annual report 2023",
            },
        )
        doc = store.get(doc_id)
        assert doc["company"] == "Apple"
        assert doc["fiscal_year"] == "2023"
        assert doc["filing_type"] == "10-K"
        assert doc["doc_description"] == "Apple annual report 2023"

    def test_create_without_metadata_returns_none_fields(self, store, sample_tree):
        """Backward compatibility: create() without metadata works, fields are None."""
        doc_id = store.create("report.pdf", sample_tree)
        doc = store.get(doc_id)
        assert doc["company"] is None
        assert doc["fiscal_year"] is None
        assert doc["filing_type"] is None
        assert doc["doc_description"] is None


class TestUpdateMetadata:
    def test_update_metadata_company(self, store, sample_tree):
        doc_id = store.create("apple_2023.pdf", sample_tree)
        result = store.update_metadata(doc_id, company="Apple Inc.")
        assert result is True
        doc = store.get(doc_id)
        assert doc["company"] == "Apple Inc."

    def test_update_metadata_multiple_fields(self, store, sample_tree):
        doc_id = store.create("apple_2023.pdf", sample_tree)
        store.update_metadata(
            doc_id,
            company="Apple",
            fiscal_year="2023",
            filing_type="10-K",
        )
        doc = store.get(doc_id)
        assert doc["company"] == "Apple"
        assert doc["fiscal_year"] == "2023"
        assert doc["filing_type"] == "10-K"

    def test_update_metadata_returns_false_for_missing_doc(self, store):
        result = store.update_metadata("pi-nonexistent", company="Apple")
        assert result is False

    def test_update_metadata_does_not_affect_other_docs(self, store, sample_tree):
        id1 = store.create("apple.pdf", sample_tree)
        id2 = store.create("google.pdf", sample_tree)
        store.update_metadata(id1, company="Apple")
        doc2 = store.get(id2)
        assert doc2["company"] is None


class TestQueryByMetadata:
    def test_query_by_company(self, store, sample_tree):
        store.create("a.pdf", sample_tree, metadata={"company": "Apple"})
        store.create("b.pdf", sample_tree, metadata={"company": "Google"})
        results = store.query_by_metadata(company="Apple")
        assert len(results) == 1
        assert results[0]["company"] == "Apple"

    def test_query_by_fiscal_year(self, store, sample_tree):
        store.create("a.pdf", sample_tree, metadata={"company": "Apple", "fiscal_year": "2023"})
        store.create("b.pdf", sample_tree, metadata={"company": "Apple", "fiscal_year": "2022"})
        results = store.query_by_metadata(fiscal_year="2023")
        assert len(results) == 1
        assert results[0]["fiscal_year"] == "2023"

    def test_query_by_filing_type(self, store, sample_tree):
        store.create("a.pdf", sample_tree, metadata={"filing_type": "10-K"})
        store.create("b.pdf", sample_tree, metadata={"filing_type": "10-Q"})
        results = store.query_by_metadata(filing_type="10-K")
        assert len(results) == 1
        assert results[0]["filing_type"] == "10-K"

    def test_query_combined_filters(self, store, sample_tree):
        store.create(
            "a.pdf",
            sample_tree,
            metadata={"company": "Apple", "fiscal_year": "2023", "filing_type": "10-K"},
        )
        store.create(
            "b.pdf",
            sample_tree,
            metadata={"company": "Apple", "fiscal_year": "2022", "filing_type": "10-K"},
        )
        store.create(
            "c.pdf",
            sample_tree,
            metadata={"company": "Google", "fiscal_year": "2023", "filing_type": "10-K"},
        )
        results = store.query_by_metadata(company="Apple", fiscal_year="2023")
        assert len(results) == 1
        assert results[0]["company"] == "Apple"
        assert results[0]["fiscal_year"] == "2023"

    def test_query_returns_empty_when_no_match(self, store, sample_tree):
        store.create("a.pdf", sample_tree, metadata={"company": "Apple"})
        results = store.query_by_metadata(company="Microsoft")
        assert results == []

    def test_query_returns_all_metadata_fields(self, store, sample_tree):
        """query_by_metadata result includes all metadata fields."""
        store.create(
            "a.pdf",
            sample_tree,
            metadata={
                "company": "Apple",
                "fiscal_year": "2023",
                "filing_type": "10-K",
                "doc_description": "Apple 10-K 2023",
            },
        )
        results = store.query_by_metadata(company="Apple")
        assert results[0]["doc_description"] == "Apple 10-K 2023"
        assert "doc_id" in results[0]
        assert "doc_name" in results[0]


class TestMetadataSqlFilter:
    def test_query_does_not_execute_sql_injection_via_company(self, store, sample_tree):
        """Parameterized query: SQL injection string treated as literal value, not SQL."""
        store.create("a.pdf", sample_tree, metadata={"company": "Apple"})
        # Injection attempt: should match nothing, not return all rows
        injection = "' OR '1'='1"
        results = store.query_by_metadata(company=injection)
        assert results == []

    def test_query_does_not_execute_sql_injection_via_fiscal_year(self, store, sample_tree):
        store.create("a.pdf", sample_tree, metadata={"fiscal_year": "2023"})
        injection = "2023; DROP TABLE documents; --"
        results = store.query_by_metadata(fiscal_year=injection)
        assert results == []

    def test_query_with_no_filters_returns_all_documents(self, store, sample_tree):
        store.create("a.pdf", sample_tree, metadata={"company": "Apple"})
        store.create("b.pdf", sample_tree, metadata={"company": "Google"})
        results = store.query_by_metadata()
        assert len(results) == 2
