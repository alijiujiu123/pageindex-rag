"""DescriptionSearcher: uses LLM to match query against document descriptions."""

from __future__ import annotations

import json
from types import SimpleNamespace

from pageindex.utils import ChatGPT_API_async, extract_json

_PROMPT_TEMPLATE = """\
You are given a list of documents with their IDs, file names, and descriptions.
Your task is to select documents that may contain information relevant to answering the user query.

Query: {query}

Documents: {documents}

Response Format:
{{
    "thinking": "<reasoning>",
    "answer": ["doc_id1", "doc_id2"]
}}
Note: return an empty list [] if no document matches.
"""


class DescriptionSearcher:
    """Searches documents by matching query against LLM-generated descriptions."""

    def __init__(self, document_store, config: SimpleNamespace):
        self._store = document_store
        self._config = config

    async def search(self, query: str) -> list[str]:
        """Return list of doc_ids whose descriptions match the query."""
        docs = self._store.query_by_metadata()
        doc_list = [
            {
                "doc_id": d["doc_id"],
                "doc_name": d["doc_name"],
                "doc_description": d.get("doc_description") or "",
            }
            for d in docs
        ]

        prompt = _PROMPT_TEMPLATE.format(
            query=query,
            documents=json.dumps(doc_list, ensure_ascii=False),
        )

        response = await ChatGPT_API_async(
            self._config.model,
            prompt,
            self._config.openai_api_key,
        )

        parsed = extract_json(response)
        return parsed.get("answer", [])
