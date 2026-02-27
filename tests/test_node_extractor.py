"""Tests for NodeContentExtractor (Issue #4)."""

from unittest.mock import MagicMock, patch

import pytest

from pageindex_rag.retrieval.node_extractor import NodeContentExtractor


# ── helpers ──────────────────────────────────────────────────────────────────

TREE = {
    "title": "Root",
    "node_id": "0000",
    "start_index": 1,
    "end_index": 10,
    "nodes": [
        {
            "title": "Chapter 1",
            "node_id": "0001",
            "start_index": 1,
            "end_index": 4,
            "nodes": [],
        },
        {
            "title": "Chapter 2",
            "node_id": "0002",
            "start_index": 5,
            "end_index": 10,
            "nodes": [],
        },
    ],
}

DOC_ID = "pi-test-doc-001"

PDF_PAGES = [(f"page {i} content", i * 100) for i in range(1, 11)]  # 10 pages


def _make_store(tree=TREE, pdf_path="/fake/doc.pdf"):
    """Return a mock DocumentStore that returns the given document."""
    store = MagicMock()
    store.get.return_value = {
        "doc_id": DOC_ID,
        "pdf_path": pdf_path,
        "tree": tree,
    }
    return store


# ── tests ─────────────────────────────────────────────────────────────────────


class TestExtractSingleNode:
    """test_extract_single_node: extract text for one node_id."""

    @patch("pageindex_rag.retrieval.node_extractor.get_page_tokens")
    def test_extract_single_node(self, mock_get_page_tokens):
        mock_get_page_tokens.return_value = PDF_PAGES

        store = _make_store()
        extractor = NodeContentExtractor(store)

        results = extractor.extract(DOC_ID, ["0001"])

        assert len(results) == 1
        assert "0001" in results
        # node 0001 covers pages 1-4
        expected = "".join(p for p, _ in PDF_PAGES[0:4])
        assert results["0001"] == expected

        store.get.assert_called_once_with(DOC_ID)
        mock_get_page_tokens.assert_called_once_with("/fake/doc.pdf")


class TestExtractMultipleNodes:
    """test_extract_multiple_nodes: extract text for multiple node_ids."""

    @patch("pageindex_rag.retrieval.node_extractor.get_page_tokens")
    def test_extract_multiple_nodes(self, mock_get_page_tokens):
        mock_get_page_tokens.return_value = PDF_PAGES

        store = _make_store()
        extractor = NodeContentExtractor(store)

        results = extractor.extract(DOC_ID, ["0001", "0002"])

        assert len(results) == 2

        expected_0001 = "".join(p for p, _ in PDF_PAGES[0:4])
        expected_0002 = "".join(p for p, _ in PDF_PAGES[4:10])

        assert results["0001"] == expected_0001
        assert results["0002"] == expected_0002

        # PDF should only be read once even for multiple nodes
        mock_get_page_tokens.assert_called_once()


class TestNodeNotFound:
    """test_node_not_found: missing node_id raises KeyError or returns empty."""

    @patch("pageindex_rag.retrieval.node_extractor.get_page_tokens")
    def test_node_not_found_raises(self, mock_get_page_tokens):
        mock_get_page_tokens.return_value = PDF_PAGES

        store = _make_store()
        extractor = NodeContentExtractor(store)

        with pytest.raises(KeyError):
            extractor.extract(DOC_ID, ["9999"])

    def test_doc_not_found_raises(self):
        store = MagicMock()
        store.get.return_value = None

        extractor = NodeContentExtractor(store)

        with pytest.raises(ValueError, match="doc_id"):
            extractor.extract("pi-nonexistent", ["0001"])
