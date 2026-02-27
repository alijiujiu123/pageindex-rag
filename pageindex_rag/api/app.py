"""FastAPI application entry point for PageIndex RAG API."""

from fastapi import FastAPI
from pageindex_rag.api.routes import documents, search, qa

app = FastAPI(title="PageIndex RAG API", version="0.1.0")

app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(search.router, prefix="/search", tags=["search"])
app.include_router(qa.router, prefix="/qa", tags=["qa"])
