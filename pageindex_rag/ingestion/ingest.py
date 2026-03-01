import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from pageindex.page_index import page_index
from pageindex.page_index_md import md_to_tree
from pageindex_rag.llm import llm_call

_executor = ThreadPoolExecutor(max_workers=2)


def _run_page_index(pdf_path: str, model: str, base_url: str) -> dict:
    """在线程池中运行 page_index（含 asyncio.run，不能直接在异步 context 中调用）。
    通过临时设置 openai 全局 base_url 让 pageindex 走 OpenRouter。
    关闭节点摘要（if_add_node_summary=no）减少约 80% token 消耗。
    """
    import openai
    old_base = openai.base_url if hasattr(openai, "base_url") else None
    try:
        openai.base_url = base_url
        return page_index(pdf_path, model=model, if_add_node_summary="no")
    finally:
        if old_base is not None:
            openai.base_url = old_base


class DocumentIngestion:
    def __init__(self, document_store, semantic_searcher, config=None):
        self.document_store = document_store
        self.semantic_searcher = semantic_searcher
        self.config = config

    async def ingest_pdf(self, pdf_path: str, metadata: dict = None) -> str:
        """
        入库 PDF 文档。
        流程：
        1. page_index_main(pdf_path) → tree_json
        2. 若 metadata 无 doc_description，调 generate_doc_description(tree) 生成
        3. DocumentStore.create(path, tree_json, metadata) → doc_id
        4. SemanticSearcher.index_document(doc_id, tree_json) 建语义索引
        返回 doc_id
        """
        # 1. PDF → tree（page_index 内部含 asyncio.run，需在线程池运行）
        pageindex_model = getattr(self.config, "pageindex_model", "gpt-4o-2024-11-20") if self.config else "gpt-4o-2024-11-20"
        base_url = getattr(self.config, "openai_base_url", "https://api.openai.com/v1") if self.config else "https://api.openai.com/v1"
        loop = asyncio.get_event_loop()
        tree_json = await loop.run_in_executor(_executor, _run_page_index, pdf_path, pageindex_model, base_url)

        # 2. 生成文档描述（如果没有提供）
        metadata = metadata or {}
        if not metadata.get("doc_description"):
            model = getattr(self.config, "model", "gpt-4o-mini") if self.config else "gpt-4o-mini"
            api_key = getattr(self.config, "openai_api_key", "") if self.config else ""
            base_url = getattr(self.config, "openai_base_url", "https://api.openai.com/v1") if self.config else "https://api.openai.com/v1"
            prompt = f"Generate a one-sentence description for this document structure that distinguishes it from other documents.\n\nDocument Structure: {tree_json}\n\nReturn only the description."
            metadata["doc_description"] = llm_call(model, prompt, api_key, base_url)

        # 3. 存储文档
        doc_id = self.document_store.create(pdf_path, tree_json, metadata)

        # 4. 建语义索引
        self.semantic_searcher.index_document(doc_id, tree_json)

        return doc_id

    async def ingest_md(self, md_path: str, metadata: dict = None) -> str:
        """
        入库 Markdown 文档。
        流程：
        1. md_to_tree(md_path) → tree_json
        2. 若 metadata 无 doc_description，调 generate_doc_description(tree) 生成
        3. DocumentStore.create(path, tree_json, metadata) → doc_id
        4. SemanticSearcher.index_document(doc_id, tree_json) 建语义索引
        返回 doc_id
        """
        # 1. MD → tree（md_to_tree 是 async）
        tree_json = await md_to_tree(md_path)

        # 2. 生成文档描述（如果没有提供）
        metadata = metadata or {}
        if not metadata.get("doc_description"):
            model = getattr(self.config, "model", "gpt-4o-mini") if self.config else "gpt-4o-mini"
            api_key = getattr(self.config, "openai_api_key", "") if self.config else ""
            base_url = getattr(self.config, "openai_base_url", "https://api.openai.com/v1") if self.config else "https://api.openai.com/v1"
            prompt = f"Generate a one-sentence description for this document structure that distinguishes it from other documents.\n\nDocument Structure: {tree_json}\n\nReturn only the description."
            metadata["doc_description"] = llm_call(model, prompt, api_key, base_url)

        # 3. 存储文档
        doc_id = self.document_store.create(md_path, tree_json, metadata)

        # 4. 建语义索引
        self.semantic_searcher.index_document(doc_id, tree_json)

        return doc_id
