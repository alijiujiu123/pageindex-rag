"""FastAPI routes for document management endpoints."""

from __future__ import annotations

import os
import tempfile
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from pydantic import BaseModel

router = APIRouter()


# ---------- Dependency factories ----------

def get_document_store():
    """Default document store dependency. Override in tests via app.dependency_overrides."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from pageindex_rag.storage.models import Base
    from pageindex_rag.storage.document_store import DocumentStore
    from pageindex_rag.config import get_config

    cfg = get_config()
    engine = create_engine(cfg.database_url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return DocumentStore(Session)


def get_ingestion():
    """Default ingestion dependency. Override in tests via app.dependency_overrides."""
    from pageindex_rag.ingestion.ingest import DocumentIngestion
    from pageindex_rag.search.semantic_search import SemanticSearcher
    from pageindex_rag.config import get_config

    cfg = get_config()
    store = get_document_store()
    searcher = SemanticSearcher(cfg)
    return DocumentIngestion(store, searcher, cfg)


# ---------- Pydantic schemas ----------

class MetadataUpdateRequest(BaseModel):
    doc_description: Optional[str] = None
    company: Optional[str] = None
    fiscal_year: Optional[str] = None
    filing_type: Optional[str] = None


# ---------- Endpoints ----------

@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    company: Optional[str] = None,
    fiscal_year: Optional[str] = None,
    filing_type: Optional[str] = None,
    doc_description: Optional[str] = None,
    ingestion=Depends(get_ingestion),
):
    """上传 PDF 或 MD 文件，入库并返回 doc_id。"""
    filename = file.filename or ""
    suffix = os.path.splitext(filename)[-1].lower()
    if suffix not in (".pdf", ".md"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .pdf and .md files are supported.",
        )

    metadata = {}
    if company:
        metadata["company"] = company
    if fiscal_year:
        metadata["fiscal_year"] = fiscal_year
    if filing_type:
        metadata["filing_type"] = filing_type
    if doc_description:
        metadata["doc_description"] = doc_description

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            doc_id = await ingestion.ingest_pdf(tmp_path, metadata)
        else:
            doc_id = await ingestion.ingest_md(tmp_path, metadata)
    finally:
        os.unlink(tmp_path)

    return {"doc_id": doc_id}


@router.get("")
def list_documents(
    company: Optional[str] = None,
    fiscal_year: Optional[str] = None,
    filing_type: Optional[str] = None,
    store=Depends(get_document_store),
):
    """列出所有文档，支持可选的元数据过滤。"""
    filters = {}
    if company:
        filters["company"] = company
    if fiscal_year:
        filters["fiscal_year"] = fiscal_year
    if filing_type:
        filters["filing_type"] = filing_type

    if filters:
        return store.query_by_metadata(**filters)
    return store.list()


@router.get("/{doc_id}")
def get_document(doc_id: str, store=Depends(get_document_store)):
    """获取单个文档详情，不存在返回 404。"""
    doc = store.get(doc_id)
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return doc


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(doc_id: str, store=Depends(get_document_store)):
    """删除文档，不存在返回 404。"""
    deleted = store.delete(doc_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")


@router.patch("/{doc_id}/metadata")
def update_metadata(
    doc_id: str,
    body: MetadataUpdateRequest,
    store=Depends(get_document_store),
):
    """更新文档元数据字段。"""
    kwargs = {k: v for k, v in body.model_dump().items() if v is not None}
    updated = store.update_metadata(doc_id, **kwargs)
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return {"doc_id": doc_id, "updated": True}
