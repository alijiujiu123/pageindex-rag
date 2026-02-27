#!/usr/bin/env python3
"""
对比优化前后的准确率。

用法：
    python scripts/compare_optimization.py --limit 10

需要：
    1. 已有 FinanceBench PDF 入库
    2. 设置 OPENAI_API_KEY 环境变量
"""

import argparse
import asyncio
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from pageindex_rag.benchmark.evaluator import AnswerEquivalenceJudge, BenchmarkEvaluator
from pageindex_rag.benchmark.financebench import FinanceBenchDataset
from pageindex_rag.config import get_config
from pageindex_rag.pipeline.rag_pipeline import RAGPipeline
from pageindex_rag.retrieval.node_extractor import NodeContentExtractor
from pageindex_rag.retrieval.tree_search import TreeSearcher
from pageindex_rag.search.metadata_search import MetadataSearcher
from pageindex_rag.search.router import DocumentSearchRouter
from pageindex_rag.search.semantic_search import SemanticSearcher
from pageindex_rag.storage.document_store import DocumentStore
from pageindex_rag.storage.models import Base


async def evaluate_baseline(config, dataset, limit, store):
    """评估 baseline（无 expert knowledge，默认权重）。"""
    semantic_searcher = SemanticSearcher(config=config)
    metadata_searcher = MetadataSearcher(document_store=store, config=config)

    router = DocumentSearchRouter(
        semantic_searcher=semantic_searcher,
        metadata_searcher=metadata_searcher,
        strategy="combined",
        # 使用等权重模拟优化前
        weights={"semantic": 1.0, "metadata": 1.0, "description": 1.0},
    )

    tree_searcher = TreeSearcher(config)
    node_extractor = NodeContentExtractor(store)

    pipeline = RAGPipeline(
        document_store=store,
        tree_searcher=tree_searcher,
        node_extractor=node_extractor,
        search_router=router,
        config=config,
    )

    judge = AnswerEquivalenceJudge(config=config)
    evaluator = BenchmarkEvaluator(pipeline, dataset, judge=judge, config=config)

    results = await evaluator.evaluate(limit=limit)
    return results


async def evaluate_optimized(config, dataset, limit, store):
    """评估优化后（注入 expert knowledge，优化权重）。"""
    semantic_searcher = SemanticSearcher(config=config)
    metadata_searcher = MetadataSearcher(document_store=store, config=config)

    router = DocumentSearchRouter(
        semantic_searcher=semantic_searcher,
        metadata_searcher=metadata_searcher,
        strategy="combined",
        # 使用优化后的权重
        weights={"semantic": 2.0, "metadata": 1.5, "description": 1.0},
    )

    tree_searcher = TreeSearcher(config)
    node_extractor = NodeContentExtractor(store)

    pipeline = RAGPipeline(
        document_store=store,
        tree_searcher=tree_searcher,
        node_extractor=node_extractor,
        search_router=router,
        config=config,
    )

    judge = AnswerEquivalenceJudge(config=config)
    evaluator = BenchmarkEvaluator(pipeline, dataset, judge=judge, config=config)

    # 注入 expert knowledge（需要修改 RAGPipeline 支持传递 expert_knowledge）
    # 这里作为演示，实际需要在调用时传递
    results = await evaluator.evaluate(limit=limit)
    return results


async def main():
    parser = argparse.ArgumentParser(description="对比优化前后的准确率")
    parser.add_argument("--limit", type=int, default=10, help="测试题目数量")
    args = parser.parse_args()

    if not os.getenv("CHATGPT_API_KEY"):
        print("错误: 需要设置 CHATGPT_API_KEY 环境变量")
        return

    config = get_config()
    engine = create_engine(config.database_url)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    store = DocumentStore(SessionLocal)
    dataset = FinanceBenchDataset()

    print(f"评估 {min(args.limit, len(dataset))} 题...")

    print("\n=== Baseline（优化前）===")
    baseline_results = await evaluate_baseline(config, dataset, args.limit, store)
    print(f"Accuracy: {baseline_results['accuracy']:.2%}")
    print(f"Passed: {baseline_results['passed']}/{baseline_results['total']}")

    print("\n=== Optimized（优化后）===")
    optimized_results = await evaluate_optimized(config, dataset, args.limit, store)
    print(f"Accuracy: {optimized_results['accuracy']:.2%}")
    print(f"Passed: {optimized_results['passed']}/{optimized_results['total']}")

    # 对比
    improvement = optimized_results['accuracy'] - baseline_results['accuracy']
    print(f"\n=== 改进 ===")
    print(f"Accuracy 提升: {improvement:+.2%}")

    if optimized_results['failed_cases']:
        print("\n=== 失败案例（优化后）===")
        for case in optimized_results['failed_cases'][:5]:
            print(f"  Q: {case['question'][:50]}...")
            print(f"  Expected: {case['expected'][:50]}...")
            print(f"  Predicted: {case['predicted'][:50]}...")
            print()


if __name__ == "__main__":
    asyncio.run(main())
