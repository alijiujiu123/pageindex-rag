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
        - "combined": merge results from all available searchers, ranked by weighted hit frequency

    Weights (for "combined" strategy):
        - semantic: Default 2.0 (most important for FinanceBench)
        - metadata: Default 1.5 (company/fiscal_year/filing_type are highly relevant)
        - description: Default 1.0 (useful fallback)
    """

    def __init__(
        self,
        description_searcher=None,
        metadata_searcher=None,
        semantic_searcher=None,
        strategy: str = "combined",
        weights: dict | None = None,
    ):
        self._description_searcher = description_searcher
        self._metadata_searcher = metadata_searcher
        self._semantic_searcher = semantic_searcher
        self._strategy = strategy
        # 默认权重优化财报场景：语义搜索 > 元数据搜索 > 描述搜索
        self._weights = weights or {
            "semantic": 2.0,
            "metadata": 1.5,
            "description": 1.0,
        }

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
        """Run all available searchers and merge results by weighted hit frequency."""
        all_results: list[tuple[str, list[str]]] = []  # (searcher_type, doc_ids)

        # Gather async searchers concurrently
        async_tasks = []
        async_indices = []

        if self._description_searcher is not None:
            async_tasks.append(self._description_searcher.search(query))
            async_indices.append(len(all_results))
            all_results.append(("description", []))  # placeholder

        if self._metadata_searcher is not None:
            async_tasks.append(self._metadata_searcher.search(query))
            async_indices.append(len(all_results))
            all_results.append(("metadata", []))  # placeholder

        if async_tasks:
            gathered = await asyncio.gather(*async_tasks)
            for idx, result in zip(async_indices, gathered):
                searcher_type, _ = all_results[idx]
                all_results[idx] = (searcher_type, result)

        # Synchronous semantic searcher
        if self._semantic_searcher is not None:
            semantic_result = self._semantic_searcher.search(query)
            all_results.append(("semantic", semantic_result))

        # Merge: apply weighted scoring, preserve first-seen order for ties
        weighted_scores: dict[str, float] = {}
        first_seen_order: list[str] = []
        seen: set[str] = set()

        for searcher_type, result_list in all_results:
            weight = self._weights.get(searcher_type, 1.0)
            for doc_id in result_list:
                if doc_id not in weighted_scores:
                    weighted_scores[doc_id] = 0.0
                    first_seen_order.append(doc_id)
                    seen.add(doc_id)
                weighted_scores[doc_id] += weight

        # Sort by weighted score descending, ties broken by first-seen order
        first_seen_index = {doc_id: i for i, doc_id in enumerate(first_seen_order)}
        ranked = sorted(
            first_seen_order,
            key=lambda d: (-weighted_scores[d], first_seen_index[d]),
        )

        return ranked
