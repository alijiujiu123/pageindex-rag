"""DocumentSearchRouter: routes search queries to appropriate searcher strategies."""

from __future__ import annotations

import asyncio
from collections import Counter


class DocumentSearchRouter:
    """Routes document search queries across description, metadata, and semantic searchers.

    Strategies:
        - "description": use DescriptionSearcher only
        - "metadata": use MetadataSearcher only
        - "semantic": use SemanticSearcher only
        - "combined": merge results from all available searchers, ranked by hit frequency
    """

    def __init__(
        self,
        description_searcher=None,
        metadata_searcher=None,
        semantic_searcher=None,
        strategy: str = "combined",
    ):
        self._description_searcher = description_searcher
        self._metadata_searcher = metadata_searcher
        self._semantic_searcher = semantic_searcher
        self._strategy = strategy

    async def search(self, query: str) -> list[str]:
        """Search documents using the configured strategy.

        Args:
            query: The user query string.

        Returns:
            List of doc_ids matching the query.
        """
        strategy = self._strategy

        if strategy == "description":
            if self._description_searcher is None:
                return []
            return await self._description_searcher.search(query)

        elif strategy == "metadata":
            if self._metadata_searcher is None:
                return []
            return await self._metadata_searcher.search(query)

        elif strategy == "semantic":
            if self._semantic_searcher is None:
                return []
            return self._semantic_searcher.search(query)

        elif strategy == "combined":
            return await self._combined_search(query)

        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

    async def _combined_search(self, query: str) -> list[str]:
        """Run all available searchers and merge results by hit frequency."""
        all_results: list[list[str]] = []

        # Gather async searchers concurrently
        async_tasks = []
        async_indices = []

        if self._description_searcher is not None:
            async_tasks.append(self._description_searcher.search(query))
            async_indices.append(len(all_results))
            all_results.append([])  # placeholder

        if self._metadata_searcher is not None:
            async_tasks.append(self._metadata_searcher.search(query))
            async_indices.append(len(all_results))
            all_results.append([])  # placeholder

        if async_tasks:
            gathered = await asyncio.gather(*async_tasks)
            for idx, result in zip(async_indices, gathered):
                all_results[idx] = result

        # Synchronous semantic searcher
        if self._semantic_searcher is not None:
            semantic_result = self._semantic_searcher.search(query)
            all_results.append(semantic_result)

        # Merge: count occurrences, preserve first-seen order for ties
        counter: Counter = Counter()
        first_seen_order: list[str] = []
        seen: set[str] = set()

        for result_list in all_results:
            for doc_id in result_list:
                counter[doc_id] += 1
                if doc_id not in seen:
                    first_seen_order.append(doc_id)
                    seen.add(doc_id)

        # Sort by count descending, ties broken by first-seen order
        first_seen_index = {doc_id: i for i, doc_id in enumerate(first_seen_order)}
        ranked = sorted(
            first_seen_order,
            key=lambda d: (-counter[d], first_seen_index[d]),
        )

        return ranked
