"""Tests for FinanceBench evaluation framework."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch


def test_load_financebench():
    """测试 FinanceBenchDataset 能正确解析数据（mock HuggingFace）。"""
    mock_rows = [
        {
            "question": "What was Apple's revenue in 2022?",
            "answer": "$394.3 billion",
            "company": "Apple",
            "fiscal_year": "2022",
            "filing_type": "10-K",
            "doc_name": "APPLE_2022_10K",
        },
        {
            "question": "What was Apple's net income in 2022?",
            "answer": "$99.8 billion",
            "company": "Apple",
            "fiscal_year": "2022",
            "filing_type": "10-K",
            "doc_name": "APPLE_2022_10K",
        },
        {
            "question": "What was Microsoft's revenue in 2022?",
            "answer": "$198.3 billion",
            "company": "Microsoft",
            "fiscal_year": "2022",
            "filing_type": "10-K",
            "doc_name": "MSFT_2022_10K",
        },
    ]

    with patch("pageindex_rag.benchmark.financebench.load_dataset") as mock_load:
        mock_load.return_value = mock_rows

        from pageindex_rag.benchmark.financebench import FinanceBenchDataset
        ds = FinanceBenchDataset(split="train")

        # 验证加载调用
        mock_load.assert_called_once_with("PatronusAI/financebench", split="train")

        # 验证长度
        assert len(ds) == 3

        # 验证 __getitem__
        item = ds[0]
        assert item["question"] == "What was Apple's revenue in 2022?"
        assert item["answer"] == "$394.3 billion"
        assert item["company"] == "Apple"
        assert item["fiscal_year"] == "2022"
        assert item["filing_type"] == "10-K"
        assert item["doc_name"] == "APPLE_2022_10K"

        # 验证 get_unique_docs（应返回2个唯一文档）
        unique_docs = ds.get_unique_docs()
        assert len(unique_docs) == 2
        doc_names = [d["doc_name"] for d in unique_docs]
        assert "APPLE_2022_10K" in doc_names
        assert "MSFT_2022_10K" in doc_names


def test_answer_equivalence_judge():
    """测试 AnswerEquivalenceJudge.judge() 返回正确布尔值（mock ChatGPT_API）。"""
    with patch("pageindex_rag.benchmark.evaluator.llm_call") as mock_api:
        from pageindex_rag.benchmark.evaluator import AnswerEquivalenceJudge

        config = MagicMock()
        config.model = "gpt-4o-2024-11-20"
        config.openai_api_key = "test-key"

        judge = AnswerEquivalenceJudge(config=config)

        # 测试等价情况（YES）
        mock_api.return_value = "YES"
        result = judge.judge("$1,500M", "$1.5B")
        assert result is True
        mock_api.assert_called_once()

        # 测试不等价情况（NO）
        mock_api.reset_mock()
        mock_api.return_value = "NO"
        result = judge.judge("$1.5B", "$2.0B")
        assert result is False
        mock_api.assert_called_once()

        # 测试响应中含有额外文字但以 YES 开头
        mock_api.reset_mock()
        mock_api.return_value = "YES, they are equivalent."
        result = judge.judge("394.3 billion", "$394.3 billion")
        assert result is True


@pytest.mark.asyncio
async def test_evaluation_report():
    """测试 BenchmarkEvaluator.evaluate() 和 generate_report() 的输出格式。"""
    # 构造 mock dataset（3题）
    mock_items = [
        {
            "question": "What was revenue?",
            "answer": "$100M",
            "doc_name": "DOC_A",
            "company": "Corp A",
            "fiscal_year": "2022",
            "filing_type": "10-K",
        },
        {
            "question": "What was net income?",
            "answer": "$20M",
            "doc_name": "DOC_A",
            "company": "Corp A",
            "fiscal_year": "2022",
            "filing_type": "10-K",
        },
        {
            "question": "What was total assets?",
            "answer": "$500M",
            "doc_name": "DOC_B",
            "company": "Corp B",
            "fiscal_year": "2022",
            "filing_type": "10-K",
        },
    ]

    mock_dataset = MagicMock()
    mock_dataset.__len__ = MagicMock(return_value=3)
    mock_dataset.__getitem__ = MagicMock(side_effect=lambda idx: mock_items[idx])

    # mock pipeline：前两题答对，第三题答错
    async def mock_query(question, doc_id=None):
        if "revenue" in question:
            return {"answer": "$100M", "sources": [{"doc_id": "pi-doc-a", "node_id": "0001"}]}
        elif "net income" in question:
            return {"answer": "$20M", "sources": [{"doc_id": "pi-doc-a", "node_id": "0002"}]}
        else:
            return {"answer": "$300M", "sources": [{"doc_id": "pi-doc-b", "node_id": "0001"}]}

    mock_pipeline = MagicMock()
    mock_pipeline.query = AsyncMock(side_effect=mock_query)

    # mock judge：revenue 和 net income 等价，total assets 不等价
    def mock_judge(predicted, expected):
        if predicted == expected:
            return True
        return False

    mock_judge_obj = MagicMock()
    mock_judge_obj.judge = MagicMock(side_effect=mock_judge)

    from pageindex_rag.benchmark.evaluator import BenchmarkEvaluator

    evaluator = BenchmarkEvaluator(
        pipeline=mock_pipeline,
        dataset=mock_dataset,
        judge=mock_judge_obj,
    )

    results = await evaluator.evaluate()

    # 验证 evaluate() 返回格式
    assert "accuracy" in results
    assert "total" in results
    assert "passed" in results
    assert "failed_cases" in results

    assert results["total"] == 3
    assert results["passed"] == 2
    assert abs(results["accuracy"] - 2 / 3) < 1e-9
    assert len(results["failed_cases"]) == 1
    assert results["failed_cases"][0]["question"] == "What was total assets?"
    assert results["failed_cases"][0]["expected"] == "$500M"
    assert results["failed_cases"][0]["predicted"] == "$300M"

    # 验证 generate_report() 输出包含 accuracy
    report = evaluator.generate_report(results)
    assert "accuracy" in report.lower() or "Accuracy" in report
    assert "2/3" in report or "66.67%" in report or "66.6" in report
