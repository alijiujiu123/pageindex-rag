"""FastAPI application entry point for PageIndex RAG API."""

from fastapi import FastAPI
from pageindex_rag.api.routes import documents

app = FastAPI(title="PageIndex RAG API", version="0.1.0")

app.include_router(documents.router, prefix="/documents", tags=["documents"])
