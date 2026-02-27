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
    """strategy='combined' 使用加权评分合并三个搜索器结果。"""
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

    # 使用默认权重：semantic=2.0, metadata=1.5, description=1.0
    # pi-001: description(1.0) + semantic(2.0) = 3.0
    # pi-002: description(1.0) + metadata(1.5) = 2.5
    # pi-003: metadata(1.5) + semantic(2.0) = 3.5
    assert len(result) == 3
    assert set(result) == {"pi-001", "pi-002", "pi-003"}
    # 按加权分降序：pi-003(3.5) > pi-001(3.0) > pi-002(2.5)
    assert result[0] == "pi-003"
    assert result[1] == "pi-001"
    assert result[2] == "pi-002"


@pytest.mark.asyncio
async def test_combined_results_with_custom_weights():
    """strategy='combined' 支持自定义权重。"""
    description_searcher = MagicMock()
    description_searcher.search = AsyncMock(return_value=["pi-001", "pi-002"])

    metadata_searcher = MagicMock()
    metadata_searcher.search = AsyncMock(return_value=["pi-002"])

    semantic_searcher = MagicMock()
    semantic_searcher.search = MagicMock(return_value=["pi-001"])

    # 使用自定义权重：description 最优先
    router = DocumentSearchRouter(
        description_searcher=description_searcher,
        metadata_searcher=metadata_searcher,
        semantic_searcher=semantic_searcher,
        strategy="combined",
        weights={"description": 3.0, "metadata": 1.0, "semantic": 1.0},
    )

    result = await router.search("test query")

    # pi-001: description(3.0) + semantic(1.0) = 4.0
    # pi-002: description(3.0) + metadata(1.0) = 4.0
    # 分数相同，按首次出现顺序
    assert len(result) == 2
    assert result == ["pi-001", "pi-002"]
