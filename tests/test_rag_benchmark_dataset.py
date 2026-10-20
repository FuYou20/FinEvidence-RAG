from pathlib import Path

import pytest

from src.scripts.build_rag_benchmark_v1 import load_jsonl, validate_cases


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data/eval/rag_benchmark_v1.jsonl"


pytestmark = pytest.mark.skipif(
    not DATASET_PATH.is_file(),
    reason="带研报原文的内部评测集未随公开仓库分发",
)


def test_rag_benchmark_has_expected_distribution():
    summary = validate_cases(load_jsonl(DATASET_PATH))

    assert summary["total"] == 50
    assert summary["task_counts"] == {
        "single_fact": 15,
        "enumeration": 10,
        "comparison": 20,
        "unanswerable": 5,
    }
    assert summary["split_counts"] == {"dev": 30, "test": 20}


def test_unanswerable_cases_define_expected_behavior():
    cases = load_jsonl(DATASET_PATH)
    unanswerable = [
        case for case in cases if case["task_type"] == "unanswerable"
    ]

    assert {case["expected_action"] for case in unanswerable} == {
        "abstain",
        "clarify",
        "external_research",
    }
    assert all(not case["answerable"] for case in unanswerable)
    assert all(not case["gold_evidence"] for case in unanswerable)


def test_comparison_cases_include_cross_document_questions():
    cases = load_jsonl(DATASET_PATH)
    comparisons = [
        case for case in cases if case["task_type"] == "comparison"
    ]

    cross_document_count = sum(
        len(case["required_doc_ids"]) >= 2 for case in comparisons
    )
    assert cross_document_count == 19


def test_single_fact_cases_define_machine_checkable_gold_facts():
    cases = load_jsonl(DATASET_PATH)
    single_fact_cases = [
        case for case in cases if case["task_type"] == "single_fact"
    ]

    assert all(case["gold_facts"] for case in single_fact_cases)
    assert all(
        isinstance(alias_group, list) and alias_group
        for case in single_fact_cases
        for alias_group in case["gold_facts"]
    )
