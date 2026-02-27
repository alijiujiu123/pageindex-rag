"""FastAPI routes for cross-document search endpoints."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

router = APIRouter()


# ---------- Dependency factories ----------

def get_search_router():
    """Default search router dependency. Override in tests via app.dependency_overrides."""
    from pageindex_rag.search.router import DocumentSearchRouter
    from pageindex_rag.search.description_search import DescriptionSearcher
    from pageindex_rag.search.metadata_search import MetadataSearcher
    from pageindex_rag.search.semantic_search import SemanticSearcher
    from pageindex_rag.storage.document_store import DocumentStore
    from pageindex_rag.config import get_config
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from pageindex_rag.storage.models import Base

    cfg = get_config()
    engine = create_engine(cfg.database_url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    store = DocumentStore(Session)

    description_searcher = DescriptionSearcher(store)
    metadata_searcher = MetadataSearcher(store)
    semantic_searcher = SemanticSearcher(cfg)

    return DocumentSearchRouter(
        description_searcher=description_searcher,
        metadata_searcher=metadata_searcher,
        semantic_searcher=semantic_searcher,
        strategy="combined",
    )


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


# ---------- Pydantic schemas ----------

class SearchRequest(BaseModel):
    query: str


class SearchResult(BaseModel):
    doc_id: str
    score: float
    metadata: dict


# ---------- Endpoints ----------

@router.post("", response_model=dict)
async def search_documents(
    request: SearchRequest,
    search_router=Depends(get_search_router),
):
    """跨文档搜索，返回匹配的文档列表。"""
    doc_ids = await search_router.search(request.query)

    results = []
    for doc_id in doc_ids:
        results.append({
            "doc_id": doc_id,
            "score": 1.0,  # TODO: implement real scoring
            "metadata": {}
        })

    return {"results": results}
