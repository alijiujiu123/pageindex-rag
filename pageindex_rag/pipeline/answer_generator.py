from pageindex_rag.config import get_config
from pageindex_rag.llm import llm_call_async


class AnswerGenerator:
    def __init__(self, config=None):
        self.config = config or get_config()

    async def generate(self, query: str, nodes_content: list) -> str:
        """
        基于检索到的节点内容生成答案。
        复用 pageindex/utils.py 中的 ChatGPT_API_async。
        """

        # 兼容旧调用：dict[node_id -> content]
        if isinstance(nodes_content, dict):
            normalized_nodes = [
                {"node_id": node_id, "content": content, "page_range": ""}
                for node_id, content in nodes_content.items()
            ]
        else:
            normalized_nodes = nodes_content or []

        # 构建上下文
        context_parts = []
        for node in normalized_nodes:
            if not isinstance(node, dict):
                continue
            content = node.get("content", "")
            node_id = node.get("node_id", "")
            page_range = node.get("page_range", "")
            context_parts.append(f"[节点 {node_id}, 页码 {page_range}]\n{content}")

        context = "\n\n".join(context_parts)

        # 针对财报场景优化的 prompt
        prompt = f"""You are analyzing a SEC financial report (10-K or 10-Q) to answer a user's question. Based on the provided document excerpts, give a precise and accurate answer.

DOCUMENT EXCERPTS:
{context}

USER QUESTION: {query}

ANSWERING GUIDELINES FOR FINANCIAL REPORTS:
1. NUMERICAL PRECISION:
   - Report numbers exactly as stated in the document (preserve units, decimals, scale)
   - Include currency symbols (e.g., $1.5 million, not 1.5)
   - Maintain percentage format (e.g., 15.3%, not 0.153)
   - Preserve scale indicators (thousands, millions, billions)

2. CONTEXTUAL ACCURACY:
   - Specify the fiscal year or reporting period for all data
   - Mention if numbers are year-over-year comparisons
   - Clarify if data is from consolidated vs. standalone statements

3. SOURCE ATTRIBUTION:
   - Reference the specific document section (e.g., "according to the Consolidated Balance Sheet")
   - Include page numbers when available

4. HANDLING AMBIGUITY:
   - If the document doesn't contain the answer, clearly state: "The provided documents do not contain information to answer this question."
   - If multiple conflicting figures exist, report all with their sources

5. FORMAT:
   - Be concise but complete
   - Use bullet points for multi-part answers
   - Highlight key numerical values

Please provide your answer:"""

        model = getattr(self.config, "model", "gpt-4o-mini")
        api_key = getattr(self.config, "openai_api_key", "")
        base_url = getattr(self.config, "openai_base_url", "https://api.openai.com/v1")
        answer = await llm_call_async(model, prompt, api_key, base_url)
        return answer
