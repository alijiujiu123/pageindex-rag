"""SemanticSearcher: ChromaDB-backed node indexing and document-level search."""

import math
import sys
import os
from types import SimpleNamespace

import chromadb

# 确保 pageindex/ 目录可被导入
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from pageindex_core.utils import get_nodes
from pageindex_rag.search.embeddings import get_embedding, get_embeddings_batch

_COLLECTION_NAME = "pageindex_nodes"


class SemanticSearcher:
    """ChromaDB-backed semantic search over indexed document nodes."""

    def __init__(self, config: SimpleNamespace):
        self._config = config
        client = chromadb.PersistentClient(path=config.chroma_persist_dir)
        self.collection = client.get_or_create_collection(_COLLECTION_NAME)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_document(self, doc_id: str, tree: dict) -> None:
        """展开树结构中所有节点，批量获取 embedding，单次 upsert 到 ChromaDB。"""
        nodes = get_nodes(tree)
        if not nodes:
            return

        node_ids = [node.get("node_id", "") for node in nodes]
        texts = [
            f"{node.get('title', '')} {node.get('summary', '')}".strip()
            for node in nodes
        ]

        # 批量获取 embedding（1次 API 调用代替 N 次串行）
        embeddings = get_embeddings_batch(
            texts,
            model=self._config.embedding_model,
            api_key=self._config.embedding_api_key,
            base_url=self._config.embedding_base_url,
        )

        chunk_ids = [f"{doc_id}#{nid}" for nid in node_ids]
        self.collection.upsert(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"doc_id": doc_id, "node_id": nid} for nid in node_ids],
        )

    def search(self, query: str, top_k: int = None) -> list[str]:
        """语义搜索，返回按 DocScore 降序排列的 doc_id 列表。"""
        if top_k is None:
            top_k = self._config.semantic_top_k

        query_embedding = get_embedding(
            query,
            model=self._config.embedding_model,
            api_key=self._config.embedding_api_key,
            base_url=self._config.embedding_base_url,
        )

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["distances", "metadatas"],
        )

        ids_list = results.get("ids", [[]])[0]
        distances_list = results.get("distances", [[]])[0]
        metadatas_list = results.get("metadatas", [[]])[0]

        if not ids_list:
            return []

        # 按 doc_id 聚合 chunk scores
        doc_chunks: dict[str, list[float]] = {}
        for distance, metadata in zip(distances_list, metadatas_list):
            doc_id = metadata["doc_id"]
            chunk_score = 1 - distance
            doc_chunks.setdefault(doc_id, []).append(chunk_score)

        # 计算 DocScore = 1/√(N+1) × Σ ChunkScore
        doc_scores: dict[str, float] = {}
        for doc_id, scores in doc_chunks.items():
            N = len(scores)
            doc_scores[doc_id] = (1 / math.sqrt(N + 1)) * sum(scores)

        # 按 DocScore 降序排列
        ranked = sorted(doc_scores.keys(), key=lambda d: doc_scores[d], reverse=True)
        return ranked

    def delete_document(self, doc_id: str) -> None:
        """删除 ChromaDB 中该 doc_id 对应的所有节点记录。"""
        self.collection.delete(where={"doc_id": doc_id})
