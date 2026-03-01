"""Configuration management for PageIndex RAG."""

import os
from types import SimpleNamespace
from dotenv import load_dotenv

load_dotenv()


def get_config(**overrides) -> SimpleNamespace:
    """Load configuration from environment variables with optional overrides."""
    # OpenRouter 优先，降级到 OpenAI
    _openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    _openai_key = os.getenv("CHATGPT_API_KEY", "")
    _use_openrouter = bool(_openrouter_key)

    defaults = {
        # LLM
        "model": os.getenv(
            "RAG_MODEL",
            "deepseek/deepseek-v3.2" if _use_openrouter else "gpt-4o-2024-11-20",
        ),
        "openai_api_key": _openrouter_key if _use_openrouter else _openai_key,
        "openai_base_url": os.getenv(
            "OPENAI_BASE_URL",
            "https://openrouter.ai/api/v1" if _use_openrouter else "https://api.openai.com/v1",
        ),
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
        "embedding_model": os.getenv(
            "EMBEDDING_MODEL",
            "openai/text-embedding-3-small" if _use_openrouter else "text-embedding-3-small",
        ),
        "embedding_api_key": os.getenv("EMBEDDING_API_KEY", _openrouter_key if _use_openrouter else _openai_key),
        "embedding_base_url": os.getenv(
            "EMBEDDING_BASE_URL",
            "https://openrouter.ai/api/v1" if _use_openrouter else "https://api.openai.com/v1",
        ),
        # Search
        "semantic_top_k": int(os.getenv("SEMANTIC_TOP_K", "20")),
        "max_search_docs": int(os.getenv("MAX_SEARCH_DOCS", "5")),
        # PageIndex（入库时使用，走 CHATGPT_API_KEY 环境变量）
        "pageindex_model": os.getenv(
            "PAGEINDEX_MODEL",
            "deepseek/deepseek-v3.2" if _use_openrouter else "gpt-4o-2024-11-20",
        ),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)
