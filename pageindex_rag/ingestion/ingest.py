import tempfile
import os

from markitdown import MarkItDown

from pageindex.page_index import page_index_main
from pageindex.page_index_md import md_to_tree
from pageindex.utils import generate_doc_description


def _convert_pdf_to_md(pdf_path: str) -> str:
    """使用 markitdown 将 PDF 转为 Markdown 字符串（纯规则，无 LLM 调用）。"""
    md = MarkItDown()
    result = md.convert(pdf_path)
    return result.text_content


class DocumentIngestion:
    def __init__(self, document_store, semantic_searcher, config=None):
        self.document_store = document_store
        self.semantic_searcher = semantic_searcher
        self.config = config

    async def ingest_pdf(self, pdf_path: str, metadata: dict = None) -> str:
        """
        入库 PDF 文档。
        流程：
        1. markitdown(pdf_path) → md_content（无 LLM，替代原 page_index_main 的 25~100次调用）
        2. 写临时文件 → md_to_tree(tmp_path) → tree_json
        3. 若 metadata 无 doc_description，调 generate_doc_description(tree) 生成（1次 LLM）
        4. DocumentStore.create(path, tree_json, metadata) → doc_id
        5. SemanticSearcher.index_document(doc_id, tree_json) 建语义索引
        返回 doc_id
        """
        # 1. PDF → MD（markitdown，纯规则，~3s）
        md_content = _convert_pdf_to_md(pdf_path)

        # 2. MD → tree（md_to_tree 接受文件路径，写入临时文件）
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as tmp:
            tmp.write(md_content)
            tmp_path = tmp.name
        try:
            tree_json = await md_to_tree(tmp_path)
        finally:
            os.unlink(tmp_path)

        # 3. 生成文档描述（如果没有提供）
        metadata = metadata or {}
        if not metadata.get("doc_description"):
            model = getattr(self.config, "model", "gpt-4o-mini") if self.config else "gpt-4o-mini"
            metadata["doc_description"] = generate_doc_description(tree_json, model)

        # 4. 存储文档
        doc_id = self.document_store.create(pdf_path, tree_json, metadata)

        # 5. 建语义索引
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
