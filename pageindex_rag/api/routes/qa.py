"""FastAPI routes for QA endpoints."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends

router = APIRouter()


# ---------- Dependency factories ----------

def get_rag_pipeline():
    """Default RAG pipeline dependency. Override in tests via app.dependency_overrides."""
    from pageindex_rag.pipeline.rag_pipeline import RAGPipeline
    from pageindex_rag.retrieval.tree_search import TreeSearcher
    from pageindex_rag.retrieval.node_extractor import NodeContentExtractor
    from pageindex_rag.search.router import DocumentSearchRouter
    from pageindex_rag.search.description_search import DescriptionSearcher
    from pageindex_rag.search.metadata_search import MetadataSearcher
    from pageindex_rag.search.semantic_search import SemanticSearcher
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from pageindex_rag.storage.models import Base
    from pageindex_rag.storage.document_store import DocumentStore
    from pageindex_rag.config import get_config

    cfg = get_config()
    engine = create_engine(cfg.database_url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    store = DocumentStore(Session)

    tree_searcher = TreeSearcher(cfg)
    node_extractor = NodeContentExtractor(store)

    description_searcher = DescriptionSearcher(store)
    metadata_searcher = MetadataSearcher(store)
    semantic_searcher = SemanticSearcher(cfg)
    search_router = DocumentSearchRouter(
        description_searcher=description_searcher,
        metadata_searcher=metadata_searcher,
        semantic_searcher=semantic_searcher,
        strategy="combined",
    )

    return RAGPipeline(
        document_store=store,
        tree_searcher=tree_searcher,
        node_extractor=node_extractor,
        search_router=search_router,
        config=cfg,
    )


# ---------- Pydantic schemas ----------

class QARequest:
    """Request schema for QA endpoint (using raw dict for flexibility)."""
    # In FastAPI, we can use dict directly or define Pydantic model
    pass


# ---------- Endpoints ----------

@router.post("")
async def answer_question(
    request: dict,
    rag_pipeline=Depends(get_rag_pipeline),
):
    """执行完整 RAG 问答，支持单文档和跨文档模式。

    请求体格式:
    {
        "query": "问题",
        "doc_id": "pi-xxx"  // 可选，不指定则跨文档
    }
    """
    query = request.get("query", "")
    doc_id = request.get("doc_id")

    result = await rag_pipeline.query(query, doc_id)
    return result
