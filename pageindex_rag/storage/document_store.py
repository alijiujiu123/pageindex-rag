"""DocumentStore: CRUD operations for PageIndex documents in PostgreSQL."""

import json
import os
import uuid

from pageindex_rag.storage.models import Document


class DocumentStore:
    """Stores and retrieves PageIndex tree structures."""

    def __init__(self, session_factory):
        self._Session = session_factory

    def create(self, pdf_path: str, tree_json: dict) -> str:
        """Store a document and return its doc_id."""
        doc_id = f"pi-{uuid.uuid4()}"
        doc_name = os.path.basename(pdf_path)
        with self._Session() as session:
            doc = Document(
                id=str(uuid.uuid4()),
                doc_id=doc_id,
                doc_name=doc_name,
                tree_json=json.dumps(tree_json),
                pdf_path=pdf_path,
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
