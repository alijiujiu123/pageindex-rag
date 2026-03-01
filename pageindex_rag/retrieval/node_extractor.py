"""NodeContentExtractor: doc_id + node_id[] -> text[]"""

from pageindex_core.utils import get_nodes, get_page_tokens


class NodeContentExtractor:
    """Extract PDF text for given node_ids from a stored document."""

    def __init__(self, document_store):
        self._store = document_store

    def extract(self, doc_id: str, node_ids: list[str]) -> dict[str, str]:
        """Return {node_id: text} for each requested node_id.

        Raises:
            ValueError: if doc_id not found in store.
            KeyError: if a node_id is not present in the document tree.
        """
        doc = self._store.get(doc_id)
        if doc is None:
            raise ValueError(f"doc_id '{doc_id}' not found")

        # Build node_id -> node map from tree
        all_nodes = get_nodes(doc["tree"])
        node_map = {n["node_id"]: n for n in all_nodes}

        # Validate all requested node_ids exist
        for nid in node_ids:
            if nid not in node_map:
                raise KeyError(nid)

        # Load PDF pages once
        pdf_pages = get_page_tokens(doc["pdf_path"])

        results = {}
        for nid in node_ids:
            node = node_map[nid]
            start = node["start_index"]
            end = node["end_index"]
            text = "".join(page_text for page_text, _ in pdf_pages[start - 1:end])
            results[nid] = text

        return results
