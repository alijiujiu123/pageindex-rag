"""FinanceBench evaluator: LLM equivalence judge and benchmark runner."""

import asyncio
from pageindex_rag.config import get_config
from pageindex.utils import ChatGPT_API


class AnswerEquivalenceJudge:
    """使用 LLM 判定两个答案是否语义等价。"""

    def __init__(self, config=None):
        self.config = config or get_config()

    def judge(self, predicted: str, expected: str) -> bool:
        """
        LLM 等价性判定，处理：
        - 数值相似性（$1.5B vs $1,500M）
        - 单位互换（百万/千万/亿）
        - 超集判定（预测包含正确答案）
        返回 True 表示等价，False 表示不等价。
        """
        prompt = f"""You are an answer equivalence judge for financial Q&A evaluation.

Determine if the "Predicted Answer" is equivalent to the "Expected Answer".

Rules:
1. Numeric equivalence: treat values like "$1.5B" and "$1,500M" as equivalent.
2. Unit conversion: handle million/billion/trillion conversions correctly.
3. Superset acceptance: if the predicted answer contains the expected answer as a correct subset, consider it equivalent.
4. Minor wording differences are acceptable as long as the factual content matches.

Expected Answer: {expected}
Predicted Answer: {predicted}

Respond with exactly one word: YES if equivalent, NO if not equivalent."""

        response = ChatGPT_API(
            model=self.config.model,
            prompt=prompt,
            api_key=self.config.openai_api_key,
        )
        return response.strip().upper().startswith("YES")


class BenchmarkEvaluator:
    """对 FinanceBench 数据集运行 RAGPipeline 并评估结果。"""

    def __init__(self, pipeline, dataset, judge=None, config=None):
        self.pipeline = pipeline
        self.dataset = dataset
        self.config = config or get_config()
        self.judge = judge or AnswerEquivalenceJudge(config=self.config)

    async def evaluate(self, limit: int = None) -> dict:
        """
        对 dataset 中每题调用 pipeline.query()，用 judge 判定是否正确。
        返回评估结果字典。
        """
        total = len(self.dataset) if limit is None else min(limit, len(self.dataset))
        passed = 0
        failed_cases = []

        for idx in range(total):
            item = self.dataset[idx]
            question = item["question"]
            expected = item["answer"]
            doc_name = item.get("doc_name", "")

            # 调用 pipeline 获取答案
            try:
                result = await self.pipeline.query(question)
                predicted = result.get("answer", "")
                doc_id = result.get("sources", [{}])[0].get("doc_id", "") if result.get("sources") else ""
            except Exception:
                predicted = ""
                doc_id = ""

            # 判定等价性
            is_correct = self.judge.judge(predicted, expected)

            if is_correct:
                passed += 1
            else:
                failed_cases.append({
                    "question": question,
                    "expected": expected,
                    "predicted": predicted,
                    "doc_id": doc_id,
                })

        accuracy = passed / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "total": total,
            "passed": passed,
            "failed_cases": failed_cases,
        }

    def generate_report(self, results: dict) -> str:
        """生成可读的评估报告文本。"""
        accuracy = results["accuracy"]
        total = results["total"]
        passed = results["passed"]
        failed = total - passed
        failed_cases = results.get("failed_cases", [])

        lines = [
            "=" * 60,
            "FinanceBench Evaluation Report",
            "=" * 60,
            f"Accuracy:  {accuracy:.2%}  ({passed}/{total})",
            f"Passed:    {passed}",
            f"Failed:    {failed}",
            "=" * 60,
        ]

        if failed_cases:
            lines.append(f"\nFailed Cases ({len(failed_cases)}):")
            for i, case in enumerate(failed_cases[:10], 1):  # 最多显示前10条
                lines.append(f"\n[{i}] Question: {case['question'][:100]}")
                lines.append(f"    Expected:  {case['expected'][:100]}")
                lines.append(f"    Predicted: {case['predicted'][:100]}")
                if case.get("doc_id"):
                    lines.append(f"    Doc ID:    {case['doc_id']}")
            if len(failed_cases) > 10:
                lines.append(f"\n... and {len(failed_cases) - 10} more failed cases.")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
