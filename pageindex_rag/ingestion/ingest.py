"""Document ingestion module using PageIndex SDK for cloud-based tree generation."""

import asyncio
import time

from pageindex import PageIndexClient

from pageindex_core.utils import generate_doc_description


class DocumentIngestion:
    """Handle PDF/MD document ingestion with PageIndex SDK."""

    def __init__(self, document_store, semantic_searcher, config=None):
        self.document_store = document_store
        self.semantic_searcher = semantic_searcher
        self.config = config
        # 初始化 PageIndex SDK 客户端
        api_key = getattr(config, "pageindex_api_key", "") if config else ""
        self.pi_client = PageIndexClient(api_key=api_key) if api_key else None

    async def ingest_pdf(self, pdf_path: str, metadata: dict = None) -> str:
        """
        入库 PDF 文档（使用 PageIndex SDK 云服务）。

        流程：
        1. pi_client.submit_document(pdf_path) → doc_id
        2. 轮询 pi_client.get_tree(doc_id) 直到 status == "completed"
        3. 若 metadata 无 doc_description，调 generate_doc_description(tree) 生成（1次 LLM）
        4. DocumentStore.create(path, tree_json, metadata) → doc_id
        5. SemanticSearcher.index_document(doc_id, tree_json) 建语义索引

        返回 doc_id
        """
        if not self.pi_client:
            raise ValueError("PageIndex API key not configured. Set PAGEINDEX_API_KEY environment variable.")

        # 1. 提交 PDF 到 PageIndex 云服务
        result = self.pi_client.submit_document(pdf_path)
        cloud_doc_id = result["doc_id"]

        # 2. 轮询等待处理完成
        tree_json = await self._poll_for_tree(cloud_doc_id)

        # 3. 生成文档描述（如果没有提供）
        metadata = metadata or {}
        if not metadata.get("doc_description"):
            model = getattr(self.config, "model", "gpt-4o-mini") if self.config else "gpt-4o-mini"
            metadata["doc_description"] = generate_doc_description(tree_json, model)

        # 4. 存储文档
        doc_id = self.document_store.create(pdf_path, tree_json, metadata)

        # 5. 建语义索引
        self.semantic_searcher.index_document(doc_id, tree_json)

        # 6. 清理云端文档（可选，节省存储）
        try:
            self.pi_client.delete_document(cloud_doc_id)
        except Exception:
            pass  # 删除失败不影响主流程

        return doc_id

    async def _poll_for_tree(self, cloud_doc_id: str, poll_interval: float = 2.0, max_wait: float = 600.0) -> list:
        """
        轮询 PageIndex 云服务直到树结构生成完成。

        Args:
            cloud_doc_id: PageIndex 云端文档 ID
            poll_interval: 轮询间隔（秒）
            max_wait: 最大等待时间（秒）

        Returns:
            树结构 JSON（list）

        Raises:
            TimeoutError: 超过最大等待时间
        """
        start_time = time.time()

        while time.time() - start_time < max_wait:
            # 在线程池中执行同步 API 调用
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.pi_client.get_tree(cloud_doc_id, node_summary=True)
            )

            status = result.get("status")
            if status == "completed":
                return result.get("result", [])
            elif status == "failed":
                raise RuntimeError(f"PageIndex tree generation failed: {result.get('error', 'Unknown error')}")

            # 等待后继续轮询
            await asyncio.sleep(poll_interval)

        raise TimeoutError(f"PageIndex tree generation timed out after {max_wait} seconds")

    async def ingest_md(self, md_path: str, metadata: dict = None) -> str:
        """
        入库 Markdown 文档。

        注意：PageIndex SDK 目前仅支持 PDF。
        对于 MD 文件，仍使用本地 md_to_tree 处理。

        流程：
        1. md_to_tree(md_path) → tree_json
        2. 若 metadata 无 doc_description，调 generate_doc_description(tree) 生成
        3. DocumentStore.create(path, tree_json, metadata) → doc_id
        4. SemanticSearcher.index_document(doc_id, tree_json) 建语义索引

        返回 doc_id
        """
        # MD 文件仍使用本地处理
        from pageindex_core.page_index_md import md_to_tree

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
