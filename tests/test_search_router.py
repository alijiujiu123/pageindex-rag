"""Tests for DocumentSearchRouter (Issue #9)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from pageindex_rag.search.router import DocumentSearchRouter


@pytest.mark.asyncio
async def test_route_to_metadata():
    """strategy='metadata' 只调用 metadata_searcher，其他搜索器不被调用。"""
    description_searcher = MagicMock()
    description_searcher.search = AsyncMock()

    metadata_searcher = MagicMock()
    metadata_searcher.search = AsyncMock(return_value=["pi-001"])

    semantic_searcher = MagicMock()
    semantic_searcher.search = MagicMock()

    router = DocumentSearchRouter(
        description_searcher=description_searcher,
        metadata_searcher=metadata_searcher,
        semantic_searcher=semantic_searcher,
        strategy="metadata",
    )

    result = await router.search("Apple 2023 10-K")

    metadata_searcher.search.assert_called_once_with("Apple 2023 10-K")
    description_searcher.search.assert_not_called()
    semantic_searcher.search.assert_not_called()
    assert result == ["pi-001"]


@pytest.mark.asyncio
async def test_route_to_description():
    """strategy='description' 只调用 description_searcher，其他搜索器不被调用。"""
    description_searcher = MagicMock()
    description_searcher.search = AsyncMock(return_value=["pi-002"])

    metadata_searcher = MagicMock()
    metadata_searcher.search = AsyncMock()

    semantic_searcher = MagicMock()
    semantic_searcher.search = MagicMock()

    router = DocumentSearchRouter(
        description_searcher=description_searcher,
        metadata_searcher=metadata_searcher,
        semantic_searcher=semantic_searcher,
        strategy="description",
    )

    result = await router.search("annual report financial highlights")

    description_searcher.search.assert_called_once_with("annual report financial highlights")
    metadata_searcher.search.assert_not_called()
    semantic_searcher.search.assert_not_called()
    assert result == ["pi-002"]


@pytest.mark.asyncio
async def test_combined_results():
    """strategy='combined' 合并三个搜索器结果，按出现次数降序排列。"""
    description_searcher = MagicMock()
    description_searcher.search = AsyncMock(return_value=["pi-001", "pi-002"])

    metadata_searcher = MagicMock()
    metadata_searcher.search = AsyncMock(return_value=["pi-002", "pi-003"])

    semantic_searcher = MagicMock()
    semantic_searcher.search = MagicMock(return_value=["pi-001", "pi-003"])

    router = DocumentSearchRouter(
        description_searcher=description_searcher,
        metadata_searcher=metadata_searcher,
        semantic_searcher=semantic_searcher,
        strategy="combined",
    )

    result = await router.search("revenue growth analysis")

    # pi-001: description + semantic = 2次
    # pi-002: description + metadata = 2次
    # pi-003: metadata + semantic = 2次
    # 三者出现次数相同，按首次出现顺序排列
    assert len(result) == 3
    # 出现2次的都在前面
    assert set(result) == {"pi-001", "pi-002", "pi-003"}
    # 按首次出现顺序：pi-001 首次出现于 description[0]，pi-002 首次出现于 description[1]，pi-003 首次出现于 metadata[1]
    assert result[0] == "pi-001"
    assert result[1] == "pi-002"
    assert result[2] == "pi-003"
