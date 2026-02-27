import asyncio
from typing import Optional


class RAGPipeline:
    def __init__(self, document_store, tree_searcher, node_extractor,
                 search_router=None, config=None):
        self.document_store = document_store
        self.tree_searcher = tree_searcher
        self.node_extractor = node_extractor
        self.search_router = search_router
        self.config = config

    @staticmethod
    def _normalize_node_ids(tree_search_result) -> list[str]:
        """Accept both TreeSearcher result dict and plain node_id list."""
        if isinstance(tree_search_result, dict):
            node_list = tree_search_result.get("node_list", [])
            return node_list if isinstance(node_list, list) else []
        if isinstance(tree_search_result, list):
            return tree_search_result
        return []

    @staticmethod
    def _extract_doc_id(search_result_item) -> str | None:
        """Accept both router result dict items and plain doc_id strings."""
        if isinstance(search_result_item, str):
            return search_result_item
        if isinstance(search_result_item, dict):
            return search_result_item.get("doc_id")
        return None

    @staticmethod
    def _build_nodes_content(doc: dict, node_ids: list[str], extracted_nodes) -> list[dict]:
        """Convert extractor output to a normalized list for answer generation."""
        if isinstance(extracted_nodes, list):
            return extracted_nodes

        if not isinstance(extracted_nodes, dict):
            return []

        tree = doc.get("tree_json") or doc.get("tree")
        node_meta_map = {}
        if tree:
            from pageindex.utils import get_nodes

            node_meta_map = {n.get("node_id"): n for n in get_nodes(tree)}

        normalized = []
        for node_id in node_ids:
            content = extracted_nodes.get(node_id, "")
            meta = node_meta_map.get(node_id, {})
            start = meta.get("start_index")
            end = meta.get("end_index")
            page_range = f"{start}-{end}" if start is not None and end is not None else ""
            normalized.append({
                "node_id": node_id,
                "content": content,
                "page_range": page_range,
            })
        return normalized

    async def query(self, query: str, doc_id: str = None) -> dict:
        """
        执行 RAG 查询。
        - 单文档（doc_id 指定）：TreeSearcher.search() → NodeContentExtractor.extract() → 答案生成
        - 多文档（doc_id=None）：SearchRouter.search() → 对每个 doc_id 做树搜索 → 汇总节点 → 答案生成
        返回: {"answer": str, "sources": list[{"doc_id", "node_id", "page_range"}]}
        """
        from pageindex_rag.pipeline.answer_generator import AnswerGenerator

        answer_generator = AnswerGenerator(config=self.config)

        if doc_id is not None:
            # 单文档模式
            doc = self.document_store.get(doc_id)
            if doc is None:
                return {"answer": "未找到指定文档。", "sources": []}

            tree = doc.get("tree_json") or doc.get("tree")
            tree_search_result = await self.tree_searcher.search(query, tree)
            node_ids = self._normalize_node_ids(tree_search_result)

            if not node_ids:
                return {"answer": "在文档中未找到相关内容。", "sources": []}

            extracted_nodes = self.node_extractor.extract(doc_id, node_ids)
            nodes_content = self._build_nodes_content(doc, node_ids, extracted_nodes)
            if not nodes_content:
                return {"answer": "在文档中未找到相关内容。", "sources": []}
            answer = await answer_generator.generate(query, nodes_content)

            sources = []
            for node in nodes_content:
                sources.append({
                    "doc_id": doc_id,
                    "node_id": node.get("node_id"),
                    "page_range": node.get("page_range")
                })

            return {"answer": answer, "sources": sources}

        else:
            # 多文档模式
            if self.search_router is None:
                return {"answer": "未配置搜索路由器，无法执行多文档查询。", "sources": []}

            search_results = await self.search_router.search(query)

            if not search_results:
                return {"answer": "未找到与查询相关的文档。", "sources": []}

            all_nodes_content = []
            all_sources = []

            for result in search_results:
                result_doc_id = self._extract_doc_id(result)
                if not result_doc_id:
                    continue
                doc = self.document_store.get(result_doc_id)
                if doc is None:
                    continue

                tree = doc.get("tree_json") or doc.get("tree")
                tree_search_result = await self.tree_searcher.search(query, tree)
                node_ids = self._normalize_node_ids(tree_search_result)

                if not node_ids:
                    continue

                extracted_nodes = self.node_extractor.extract(result_doc_id, node_ids)
                nodes_content = self._build_nodes_content(doc, node_ids, extracted_nodes)
                if not nodes_content:
                    continue
                all_nodes_content.extend(nodes_content)

                for node in nodes_content:
                    all_sources.append({
                        "doc_id": result_doc_id,
                        "node_id": node.get("node_id"),
                        "page_range": node.get("page_range")
                    })

            if not all_nodes_content:
                return {"answer": "在相关文档中未找到具体内容。", "sources": []}

            answer = await answer_generator.generate(query, all_nodes_content)
            return {"answer": answer, "sources": all_sources}
