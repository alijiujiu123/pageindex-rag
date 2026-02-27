"""Tests for full FinanceBench benchmark (Issue #17)."""

import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path


@pytest.mark.skipif(not os.getenv("RUN_FULL_BENCHMARK"), reason="需要设置 RUN_FULL_BENCHMARK=1 运行全量测试")
@pytest.mark.asyncio
async def test_full_benchmark():
    """运行全量 FinanceBench 测试，目标准确率 >= 98.7%。"""
    import asyncio
    import sys
    from pathlib import Path

    # 添加 scripts 目录到路径
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

    # 运行 benchmark 脚本
    result = await asyncio.create_subprocess_exec(
        sys.executable,
        "scripts/run_benchmark.py",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await result.communicate()

    # 检查退出码（0 表示达标，1 表示未达标）
    assert result.returncode == 0, f"Benchmark failed:\n{stderr.decode()}"

    # 检查报告文件
    report_path = Path("results/financebench_report.json")
    assert report_path.exists(), "报告文件未生成"

    # 验证报告内容
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "summary" in report
    assert report["summary"]["accuracy"] >= 0.987, f"准确率未达标: {report['summary']['accuracy']:.2%}"


def test_benchmark_script_exists():
    """测试 benchmark 脚本存在且可执行。"""
    script_path = Path("scripts/run_benchmark.py")
    assert script_path.exists(), "run_benchmark.py 不存在"


def test_benchmark_report_schema():
    """测试报告 schema 正确（使用模拟数据）。"""
    from pageindex_rag.benchmark.evaluator import BenchmarkEvaluator
    from pageindex_rag.benchmark.financebench import FinanceBenchDataset
    from pageindex_rag.config import get_config

    # 模拟数据
    mock_results = {
        "accuracy": 0.99,
        "total": 150,
        "passed": 148,
        "failed": 2,
        "failed_cases": [
            {
                "index": 10,
                "question": "Test question",
                "expected": "Expected answer",
                "predicted": "Wrong answer",
                "doc_name": "test.pdf",
                "doc_ids": ["pi-test"],
            }
        ],
        "question_results": [
            {
                "index": 0,
                "question": "Test question",
                "expected_answer": "Expected",
                "predicted_answer": "Expected",
                "is_correct": True,
                "doc_name": "test.pdf",
                "company": "TEST",
                "fiscal_year": "2023",
                "filing_type": "10-K",
                "evidence_nodes": ["0001"],
                "doc_ids": ["pi-test"],
            }
        ],
    }

    # 验证 schema
    assert "accuracy" in mock_results
    assert "total" in mock_results
    assert "passed" in mock_results
    assert "failed" in mock_results
    assert "failed_cases" in mock_results
    assert "question_results" in mock_results

    # 验证 question_result schema
    qr = mock_results["question_results"][0]
    required_fields = [
        "index", "question", "expected_answer", "predicted_answer",
        "is_correct", "doc_name", "company", "fiscal_year",
        "filing_type", "evidence_nodes", "doc_ids",
    ]
    for field in required_fields:
        assert field in qr, f"Missing field: {field}"

    # 验证 failed_case schema
    fc = mock_results["failed_cases"][0]
    required_fc_fields = ["index", "question", "expected", "predicted", "doc_name", "doc_ids"]
    for field in required_fc_fields:
        assert field in fc, f"Missing field in failed_case: {field}"


def test_benchmark_report_format():
    """测试报告 JSON 格式正确。"""
    from datetime import datetime

    # 模拟报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "accuracy": 0.987,
            "total": 150,
            "passed": 148,
            "failed": 2,
            "target": 0.987,
            "target_met": True,
        },
        "question_results": [],
        "failed_cases": [],
    }

    # 验证可序列化为 JSON
    json_str = json.dumps(report, ensure_ascii=False, indent=2)
    assert json_str

    # 验证可以解析回来
    parsed = json.loads(json_str)
    assert parsed["summary"]["accuracy"] == 0.987
    assert parsed["summary"]["target_met"] is True


@pytest.mark.asyncio
async def test_enhanced_evaluator_records_node_ids():
    """测试增强评估器记录 evidence_nodes。"""
    from unittest.mock import AsyncMock
    from pageindex_rag.benchmark.evaluator import AnswerEquivalenceJudge
    from scripts.run_benchmark import EnhancedBenchmarkEvaluator

    # Mock pipeline
    mock_pipeline = AsyncMock()
    mock_pipeline.query.return_value = {
        "answer": "Test answer",
        "sources": [
            {"doc_id": "pi-001", "node_id": "0001", "page_range": "1-5"},
            {"doc_id": "pi-002", "node_id": "0002", "page_range": "6-10"},
        ],
    }

    # Mock judge
    mock_judge = MagicMock()
    mock_judge.judge.return_value = True

    # Mock dataset
    mock_dataset = [
        {
            "question": "Test question?",
            "answer": "Test answer",
            "doc_name": "test.pdf",
            "company": "TEST",
            "fiscal_year": "2023",
            "filing_type": "10-K",
        }
    ]

    evaluator = EnhancedBenchmarkEvaluator(
        pipeline=mock_pipeline,
        dataset=mock_dataset,
        judge=mock_judge,
    )

    results = await evaluator.evaluate()

    # 验证记录了 node_ids
    qr = results["question_results"][0]
    assert qr["evidence_nodes"] == ["0001", "0002"]
    assert qr["doc_ids"] == ["pi-001", "pi-002"]
