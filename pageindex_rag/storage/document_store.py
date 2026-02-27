"""DocumentStore: CRUD operations for PageIndex documents in PostgreSQL."""

from __future__ import annotations

import json
import os
import uuid

from pageindex_rag.storage.models import Document

_METADATA_FIELDS = ("doc_description", "company", "fiscal_year", "filing_type")


class DocumentStore:
    """Stores and retrieves PageIndex tree structures."""

    def __init__(self, session_factory):
        self._Session = session_factory

    def create(self, pdf_path: str, tree_json: dict, metadata: dict = None) -> str:
        """Store a document and return its doc_id."""
        doc_id = f"pi-{uuid.uuid4()}"
        doc_name = os.path.basename(pdf_path)
        meta = metadata or {}
        with self._Session() as session:
            doc = Document(
                id=str(uuid.uuid4()),
                doc_id=doc_id,
                doc_name=doc_name,
                tree_json=json.dumps(tree_json),
                pdf_path=pdf_path,
                doc_description=meta.get("doc_description"),
                company=meta.get("company"),
                fiscal_year=meta.get("fiscal_year"),
                filing_type=meta.get("filing_type"),
            )
            session.add(doc)
            session.commit()
        return doc_id

    def get(self, doc_id: str) -> dict | None:
        """Return document dict or None if not found."""
        with self._Session() as session:
            doc = session.query(Document).filter_by(doc_id=doc_id).first()
            if doc is None:
                return None
            return {
                "doc_id": doc.doc_id,
                "doc_name": doc.doc_name,
                "pdf_path": doc.pdf_path,
                "tree": json.loads(doc.tree_json),
                "created_at": doc.created_at,
                "doc_description": doc.doc_description,
                "company": doc.company,
                "fiscal_year": doc.fiscal_year,
                "filing_type": doc.filing_type,
            }

    def list(self) -> list[dict]:
        """Return summary list of all documents (no tree_json)."""
        with self._Session() as session:
            docs = session.query(Document).all()
            return [
                {
                    "doc_id": d.doc_id,
                    "doc_name": d.doc_name,
                    "created_at": d.created_at,
                    "company": d.company,
                    "fiscal_year": d.fiscal_year,
                    "filing_type": d.filing_type,
                }
                for d in docs
            ]

    def delete(self, doc_id: str) -> bool:
        """Delete a document. Returns True if deleted, False if not found."""
        with self._Session() as session:
            doc = session.query(Document).filter_by(doc_id=doc_id).first()
            if doc is None:
                return False
            session.delete(doc)
            session.commit()
            return True

    def update_metadata(self, doc_id: str, **kwargs) -> bool:
        """Update metadata fields for a document. Returns True if updated, False if not found."""
        with self._Session() as session:
            doc = session.query(Document).filter_by(doc_id=doc_id).first()
            if doc is None:
                return False
            for field in _METADATA_FIELDS:
                if field in kwargs:
                    setattr(doc, field, kwargs[field])
            session.commit()
            return True

    def query_by_metadata(self, **filters) -> list[dict]:
        """Return documents matching all provided metadata filters.

        Uses SQLAlchemy ORM parameterized queries to prevent SQL injection.
        No filters → returns all documents.
        """
        with self._Session() as session:
            query = session.query(Document)
            for field, value in filters.items():
                if field in _METADATA_FIELDS:
                    query = query.filter(getattr(Document, field) == value)
            docs = query.all()
            return [
                {
                    "doc_id": d.doc_id,
                    "doc_name": d.doc_name,
                    "created_at": d.created_at,
                    "doc_description": d.doc_description,
                    "company": d.company,
                    "fiscal_year": d.fiscal_year,
                    "filing_type": d.filing_type,
                }
                for d in docs
            ]
