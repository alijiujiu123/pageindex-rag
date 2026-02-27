"""MetadataSearcher: LLM-powered metadata filter search for PageIndex RAG."""

from __future__ import annotations

from types import SimpleNamespace

from pageindex.utils import ChatGPT_API_async, extract_json

_METADATA_PROMPT_TEMPLATE = """从用户查询中提取文档元数据筛选条件。
可用字段: company (str), fiscal_year (str), filing_type (str)
查询: {query}
返回 JSON: {{"company": "...", "fiscal_year": "...", "filing_type": "..."}}
（无法确定的字段设为 null）"""


class MetadataSearcher:
    """Searches documents by LLM-extracted metadata conditions."""

    def __init__(self, document_store, config: SimpleNamespace):
        self._store = document_store
        self._config = config

    async def search(self, query: str) -> list[str]:
        """Extract metadata filters from query via LLM and return matching doc_id list."""
        prompt = _METADATA_PROMPT_TEMPLATE.format(query=query)
        try:
            raw = await ChatGPT_API_async(
                self._config.model, prompt, self._config.openai_api_key
            )
            filters_raw: dict = extract_json(raw)
        except Exception:
            return []

        if not filters_raw:
            return []

        # Filter out null/None values
        filters = {k: v for k, v in filters_raw.items() if v is not None}

        docs = self._store.query_by_metadata(**filters)
        return [doc["doc_id"] for doc in docs]
