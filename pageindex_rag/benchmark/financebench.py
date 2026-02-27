"""FinanceBench dataset loader for PageIndex RAG evaluation."""

from datasets import load_dataset


class FinanceBenchDataset:
    """加载并提供 FinanceBench 数据集访问接口。"""

    def __init__(self, split="train"):
        """从 HuggingFace 加载 PatronusAI/financebench 数据集。"""
        ds = load_dataset("PatronusAI/financebench", split=split)
        self._data = list(ds)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx) -> dict:
        """返回标准化的题目字典。"""
        row = self._data[idx]
        return {
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "company": row.get("company", ""),
            "fiscal_year": row.get("fiscal_year", ""),
            "filing_type": row.get("filing_type", ""),
            "doc_name": row.get("doc_name", ""),
        }

    def get_unique_docs(self) -> list:
        """返回需要入库的唯一文档列表（按 doc_name 去重）。"""
        seen = set()
        unique = []
        for item in self._data:
            doc_name = item.get("doc_name", "")
            if doc_name and doc_name not in seen:
                seen.add(doc_name)
                unique.append({
                    "doc_name": doc_name,
                    "company": item.get("company", ""),
                    "fiscal_year": item.get("fiscal_year", ""),
                    "filing_type": item.get("filing_type", ""),
                })
        return unique
