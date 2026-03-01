"""TreeSearcher: LLM-based tree node search for Issue #5."""

import json
import logging
from types import SimpleNamespace

from pageindex_core.utils import extract_json
from pageindex_rag.llm import llm_call_async

logger = logging.getLogger(__name__)


# SEC 财报领域知识模板，用于 FinanceBench 等财报问答场景
SEC_FINANCIAL_REPORT_EXPERT = """
SEC 10-K/10-Q Financial Report Structure:

KEY SECTIONS FOR COMMON QUERIES:
- Financial Statements (Balance Sheet, Income Statement, Cash Flow Statement) → numerical metrics, ratios
- Management's Discussion and Analysis (MD&A) → narrative analysis, trends, explanations
- Notes to Consolidated Financial Statements → detailed accounting policies, breakdowns
- Business Overview → company description, operations, markets
- Risk Factors → potential risks and uncertainties
- Legal Proceedings → litigation, regulatory issues

COMMON FINANCIAL METRICS:
Revenue, Net Income, EBITDA, Operating Cash Flow, Total Assets, Total Debt,
Shareholders' Equity, Earnings Per Share (EPS), Free Cash Flow, Working Capital

QUERY PATTERNS:
- "What was [company]'s [metric] in [year]?" → Financial Statements
- "How did [metric] change from [year] to [year]?" → MD&A + Financial Statements comparison
- "What caused the change in [metric]?" → MD&A explanation
- "What is the company's main business?" → Business Overview
- "What are the main risks?" → Risk Factors
"""


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
        # 针对财报场景优化的 prompt
        prompt = f"""You are analyzing a SEC financial report (10-K or 10-Q). Your task is to identify which document sections contain the answer to the user's query.

USER QUERY: {query}

DOCUMENT TREE STRUCTURE:
{json.dumps(tree_structure, ensure_ascii=False, indent=2)}

ANALYSIS INSTRUCTIONS:
1. Carefully read the user query and identify key information needs (company name, fiscal year, data type, specific metric)
2. Review the document tree structure to understand the report organization
3. Select nodes that are MOST LIKELY to contain the answer based on:
   - Section titles (e.g., "Financial Statements", "Management's Discussion", "Notes to Consolidated Financial Statements")
   - Query keywords matching (e.g., "revenue", "net income", "cash flow", "debt", "assets")
   - Hierarchical relevance (parent nodes may contain the answer if child nodes are relevant)

IMPORTANT RULES:
- Be thorough but selective: include all potentially relevant nodes, but avoid obviously irrelevant ones
- For numerical queries: prioritize sections containing financial statements, notes, and MD&A
- For qualitative queries: consider Management's Discussion, Business description, Risk Factors
- Include parent nodes when the answer might span multiple child sections
- Return node_ids as a list, e.g., ["0001", "0002", "0003"]

Reply in JSON format: {{"thinking": "your reasoning process...", "node_list": ["node_id1", "node_id2", ...]}}"""

        combined = " ".join(filter(None, [expert_knowledge, preference]))
        if combined:
            prompt += f"\nExpert Knowledge of relevant sections: {combined}"

        try:
            response = await llm_call_async(
                self.config.model,
                prompt,
                self.config.openai_api_key,
                self.config.openai_base_url,
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
