"""Configuration management for PageIndex RAG."""

import os
from types import SimpleNamespace
from dotenv import load_dotenv

load_dotenv()


def get_config(**overrides) -> SimpleNamespace:
    """Load configuration from environment variables with optional overrides."""
    defaults = {
        # LLM
        "model": os.getenv("RAG_MODEL", "gpt-4o-2024-11-20"),
        "openai_api_key": os.getenv("CHATGPT_API_KEY", ""),
        # Database
        "database_url": os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/pageindex_rag",
        ),
        # ChromaDB
        "chroma_host": os.getenv("CHROMA_HOST", "localhost"),
        "chroma_port": int(os.getenv("CHROMA_PORT", "8000")),
        "chroma_persist_dir": os.getenv("CHROMA_PERSIST_DIR", "./chroma_data"),
        # Embedding
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        # Search
        "semantic_top_k": int(os.getenv("SEMANTIC_TOP_K", "20")),
        "max_search_docs": int(os.getenv("MAX_SEARCH_DOCS", "5")),
        # PageIndex
        "pageindex_model": os.getenv("PAGEINDEX_MODEL", "gpt-4o-2024-11-20"),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)
