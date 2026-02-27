from pageindex_rag.config import get_config
from pageindex.utils import ChatGPT_API_async


class AnswerGenerator:
    def __init__(self, config=None):
        self.config = config or get_config()

    async def generate(self, query: str, nodes_content: list) -> str:
        """
        基于检索到的节点内容生成答案。
        复用 pageindex/utils.py 中的 ChatGPT_API_async。
        """

        # 构建上下文
        context_parts = []
        for node in nodes_content:
            content = node.get("content", "")
            node_id = node.get("node_id", "")
            page_range = node.get("page_range", "")
            context_parts.append(f"[节点 {node_id}, 页码 {page_range}]\n{content}")

        context = "\n\n".join(context_parts)

        prompt = f"""请根据以下文档内容回答问题。

文档内容：
{context}

问题：{query}

请基于文档内容给出准确、详细的回答。如果文档内容不足以回答问题，请说明。"""

        messages = [{"role": "user", "content": prompt}]

        model = getattr(self.config, "model", "gpt-4o-mini")
        answer = await ChatGPT_API_async(messages, model=model)
        return answer
