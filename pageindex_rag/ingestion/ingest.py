import uuid
from typing import Optional

from pageindex.page_index import page_index_main
from pageindex.page_index_md import md_to_tree
from pageindex.utils import generate_doc_description


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
        # 1. PDF → tree
        tree_json = page_index_main(pdf_path)

        # 2. 生成文档描述（如果没有提供）
        metadata = metadata or {}
        if not metadata.get("doc_description"):
            model = getattr(self.config, "model", "gpt-4o-mini") if self.config else "gpt-4o-mini"
            metadata["doc_description"] = generate_doc_description(tree_json, model)

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
            metadata["doc_description"] = generate_doc_description(tree_json, model)

        # 3. 存储文档
        doc_id = self.document_store.create(md_path, tree_json, metadata)

        # 4. 建语义索引
        self.semantic_searcher.index_document(doc_id, tree_json)

        return doc_id
