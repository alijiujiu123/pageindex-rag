#!/usr/bin/env python3
"""
FinanceBench 全量基准测试脚本。

运行 150 题测试，生成完整报告到 results/financebench_report.json。

用法：
    python scripts/run_benchmark.py [--limit N] [--output PATH]

环境变量：
    CHATGPT_API_KEY: OpenAI API Key
    DATABASE_URL: PostgreSQL 连接字符串
    CHROMA_PERSIST_DIR: ChromaDB 持久化目录
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from pageindex_rag.benchmark.evaluator import AnswerEquivalenceJudge, BenchmarkEvaluator
from pageindex_rag.benchmark.financebench import FinanceBenchDataset
from pageindex_rag.config import get_config
from pageindex_rag.ingestion.ingest import DocumentIngestion
from pageindex_rag.pipeline.rag_pipeline import RAGPipeline
from pageindex_rag.retrieval.node_extractor import NodeContentExtractor
from pageindex_rag.retrieval.tree_search import TreeSearcher
from pageindex_rag.search.metadata_search import MetadataSearcher
from pageindex_rag.search.router import DocumentSearchRouter
from pageindex_rag.search.semantic_search import SemanticSearcher
from pageindex_rag.storage.document_store import DocumentStore


class EnhancedBenchmarkEvaluator(BenchmarkEvaluator):
    """增强的评估器，记录每题的详细信息。"""

    async def evaluate(self, limit: int = None) -> dict:
        """
        对 dataset 中每题调用 pipeline.query()，用 judge 判定是否正确。
        返回包含每题详细信息的评估结果。
        """
        total = len(self.dataset) if limit is None else min(limit, len(self.dataset))
        passed = 0
        failed_cases = []
        question_results = []

        for idx in range(total):
            item = self.dataset[idx]
            question = item["question"]
            expected = item["answer"]
            doc_name = item.get("doc_name", "")
            company = item.get("company", "")
            fiscal_year = item.get("fiscal_year", "")
            filing_type = item.get("filing_type", "")

            print(f"[{idx + 1}/{total}] {company} {fiscal_year} {filing_type}: {question[:50]}...")

            # 调用 pipeline 获取答案
            try:
                result = await self.pipeline.query(question)
                predicted = result.get("answer", "")
                sources = result.get("sources", [])
                doc_ids = [s.get("doc_id", "") for s in sources]
                node_ids = [s.get("node_id", "") for s in sources]
            except Exception as e:
                print(f"  ❌ Error: {e}")
                predicted = ""
                doc_ids = []
                node_ids = []

            # 判定等价性
            is_correct = self.judge.judge(predicted, expected)

            question_result = {
                "index": idx,
                "question": question,
                "expected_answer": expected,
                "predicted_answer": predicted,
                "is_correct": is_correct,
                "doc_name": doc_name,
                "company": company,
                "fiscal_year": fiscal_year,
                "filing_type": filing_type,
                "evidence_nodes": node_ids,
                "doc_ids": doc_ids,
            }
            question_results.append(question_result)

            if is_correct:
                passed += 1
                print(f"  ✓ Correct")
            else:
                failed_cases.append({
                    "index": idx,
                    "question": question,
                    "expected": expected,
                    "predicted": predicted,
                    "doc_name": doc_name,
                    "doc_ids": doc_ids,
                })
                print(f"  ✗ Wrong")

        accuracy = passed / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "failed_cases": failed_cases,
            "question_results": question_results,
        }

    def generate_detailed_report(self, results: dict) -> dict:
        """生成详细的 JSON 报告。"""
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "accuracy": results["accuracy"],
                "total": results["total"],
                "passed": results["passed"],
                "failed": results["failed"],
                "target": 0.987,
                "target_met": results["accuracy"] >= 0.987,
            },
            "question_results": results["question_results"],
            "failed_cases": results["failed_cases"],
        }


async def main():
    parser = argparse.ArgumentParser(description="FinanceBench 全量基准测试")
    parser.add_argument("--limit", type=int, default=None, help="限制测试题目数量")
    parser.add_argument("--output", type=str, default="results/financebench_report.json", help="输出报告路径")
    args = parser.parse_args()

    # 检查环境变量
    if not os.getenv("CHATGPT_API_KEY"):
        print("错误: 需要设置 CHATGPT_API_KEY 环境变量")
        sys.exit(1)

    print("=" * 60)
    print("FinanceBench 全量基准测试")
    print("=" * 60)

    # 初始化
    config = get_config()
    store = DocumentStore(config=config)

    # 搜索器
    semantic_searcher = SemanticSearcher(config=config)
    semantic_searcher.load_index()

    metadata_searcher = MetadataSearcher(store=store)

    # 路由器（使用优化后的权重）
    router = DocumentSearchRouter(
        semantic_searcher=semantic_searcher,
        metadata_searcher=metadata_searcher,
        strategy="combined",
        weights={"semantic": 2.0, "metadata": 1.5, "description": 1.0},
    )

    # 树搜索器
    tree_searcher = TreeSearcher(config)

    # 节点提取器
    node_extractor = NodeContentExtractor(store=store)

    # Pipeline
    pipeline = RAGPipeline(
        document_store=store,
        tree_searcher=tree_searcher,
        node_extractor=node_extractor,
        search_router=router,
        config=config,
    )

    # 数据集
    dataset = FinanceBenchDataset()
    limit = args.limit or len(dataset)
    print(f"\n数据集大小: {len(dataset)} 题")
    print(f"测试题目数: {limit} 题")
    print(f"目标准确率: 98.7%")

    # 评估器
    judge = AnswerEquivalenceJudge(config=config)
    evaluator = EnhancedBenchmarkEvaluator(pipeline, dataset, judge=judge, config=config)

    # 运行评估
    print("\n开始评估...")
    print("-" * 60)
    results = await evaluator.evaluate(limit=limit)

    # 生成报告
    print("\n" + "=" * 60)
    print("评估完成!")
    print("=" * 60)
    print(f"准确率: {results['accuracy']:.2%}  ({results['passed']}/{results['total']})")
    print(f"目标:   98.7%")
    print(f"结果:   {'✓ 达标' if results['accuracy'] >= 0.987 else '✗ 未达标'}")

    # 保存 JSON 报告
    report = evaluator.generate_detailed_report(results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已保存到: {output_path}")

    # 打印失败案例
    if results["failed_cases"]:
        print(f"\n失败案例 ({len(results['failed_cases'])} 条):")
        for case in results["failed_cases"][:10]:
            print(f"\n  [{case['index']}] {case['question'][:80]}")
            print(f"      期望: {case['expected'][:80]}")
            print(f"      预测: {case['predicted'][:80]}")
        if len(results["failed_cases"]) > 10:
            print(f"\n  ... 还有 {len(results['failed_cases']) - 10} 条失败案例")

    # 按公司统计
    company_stats = {}
    for qr in results["question_results"]:
        company = qr["company"]
        if company not in company_stats:
            company_stats[company] = {"total": 0, "passed": 0}
        company_stats[company]["total"] += 1
        if qr["is_correct"]:
            company_stats[company]["passed"] += 1

    print("\n按公司统计:")
    for company, stats in sorted(company_stats.items()):
        acc = stats["passed"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {company}: {acc:.1%} ({stats['passed']}/{stats['total']})")

    # 退出码
    sys.exit(0 if results["accuracy"] >= 0.987 else 1)


if __name__ == "__main__":
    asyncio.run(main())
