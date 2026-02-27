"""TreeSearcher: LLM-based tree node search for Issue #5."""

import json
import logging
from types import SimpleNamespace

from pageindex.utils import ChatGPT_API_async, extract_json

logger = logging.getLogger(__name__)


class TreeSearcher:
    """Uses LLM to identify relevant nodes in a document tree structure."""

    def __init__(self, config: SimpleNamespace):
        self.config = config

    async def search(
        self,
        query: str,
        tree_structure: dict,
        expert_knowledge: str = "",
        preference: str = "",
    ) -> dict:
        """
        Search for relevant nodes in the tree structure using LLM reasoning.

        Args:
            query: The user's question.
            tree_structure: The document tree (PageIndex format).
            expert_knowledge: Optional expert knowledge about relevant sections.
            preference: Optional preference/guidance for the search.

        Returns:
            {"thinking": str, "node_list": list[str]}
        """
        prompt = (
            f"You are given a query and the tree structure of a document.\n"
            f"You need to find all nodes that are likely to contain the answer.\n"
            f"Query: {query}\n"
            f"Document tree structure: {json.dumps(tree_structure, ensure_ascii=False)}\n"
            f'Reply in JSON: {{"thinking": ..., "node_list": [node_id1, ...]}}'
        )

        combined = " ".join(filter(None, [expert_knowledge, preference]))
        if combined:
            prompt += f"\nExpert Knowledge of relevant sections: {combined}"

        try:
            response = await ChatGPT_API_async(
                self.config.model,
                prompt,
                self.config.openai_api_key,
            )
            parsed = extract_json(response)
            if not parsed or "node_list" not in parsed:
                return {"thinking": "", "node_list": []}
            return {
                "thinking": parsed.get("thinking", ""),
                "node_list": parsed.get("node_list", []),
            }
        except Exception as e:
            logger.error(f"TreeSearcher search failed: {e}")
            return {"thinking": "", "node_list": []}
