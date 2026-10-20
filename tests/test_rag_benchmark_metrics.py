import pytest

from src.evaluation.rag_benchmark_metrics import evaluate_rag_benchmark


def make_prediction(query_id, **overrides):
    prediction = {
        "query_id": query_id,
        "predicted_action": "answer_with_citations",
        "answer_text": "",
        "cited_chunk_ids": [],
        "cited_doc_ids": [],
        "predicted_items": [],
    }
    prediction.update(overrides)
    return prediction


def test_single_fact_uses_case_accuracy_and_fact_recall():
    cases = [
        {
            "query_id": "fact_1",
            "task_type": "single_fact",
            "required_doc_ids": ["doc_1"],
            "gold_evidence": [{"source_chunk_id": "chunk_1"}],
            "gold_facts": [["50亿元", "50亿元人民币"], ["83.02%"]],
        }
    ]
    predictions = [
        make_prediction(
            "fact_1",
            answer_text="公司计划增资50亿元，但当前答案没有给出持股比例。",
            cited_chunk_ids=["chunk_1"],
            cited_doc_ids=["doc_1"],
        )
    ]

    result = evaluate_rag_benchmark(cases, predictions)

    assert result["single_fact"]["accuracy"] == 0.0
    assert result["single_fact"]["fact_recall"] == 0.5
    assert result["single_fact"]["evidence_chunk_recall"] == 1.0


def test_enumeration_reports_item_metrics_and_citation_accuracy():
    cases = [
        {
            "query_id": "enum_1",
            "task_type": "enumeration",
            "required_doc_ids": ["doc_1"],
            "gold_evidence": [
                {"source_chunk_id": "chunk_a"},
                {"source_chunk_id": "chunk_b"},
            ],
            "gold_items": [
                {"item": "渠道覆盖", "evidence_chunk_ids": ["chunk_a"]},
                {"item": "本土化运营", "evidence_chunk_ids": ["chunk_b"]},
            ],
        }
    ]
    predictions = [
        make_prediction(
            "enum_1",
            cited_doc_ids=["doc_1"],
            cited_chunk_ids=["chunk_a"],
            predicted_items=[
                {"item": "渠道覆盖", "cited_chunk_ids": ["chunk_a"]},
                {"item": "价格上涨", "cited_chunk_ids": ["chunk_x"]},
            ],
        )
    ]

    result = evaluate_rag_benchmark(cases, predictions)["enumeration"]

    assert result["item_precision"] == 0.5
    assert result["item_recall"] == 0.5
    assert result["item_f1"] == 0.5
    assert result["complete_answer"] == 0.0
    assert result["item_citation_accuracy"] == 1.0


def test_enumeration_can_use_semantic_judge_matches():
    cases = [
        {
            "query_id": "enum_1",
            "task_type": "enumeration",
            "required_doc_ids": ["doc_1"],
            "gold_evidence": [{"source_chunk_id": "chunk_a"}],
            "gold_items": [
                {
                    "item": "细胞学诊断平台",
                    "evidence_chunk_ids": ["chunk_a"],
                }
            ],
        }
    ]
    predictions = [
        make_prediction(
            "enum_1",
            cited_doc_ids=["doc_1"],
            cited_chunk_ids=["chunk_a"],
            predicted_items=[
                {
                    "item": "细胞学诊断技术平台",
                    "cited_chunk_ids": ["chunk_a"],
                }
            ],
            enumeration_matches=[
                {"gold_index": 0, "predicted_index": 0}
            ],
        )
    ]

    result = evaluate_rag_benchmark(cases, predictions)["enumeration"]

    assert result["item_precision"] == 1.0
    assert result["item_recall"] == 1.0
    assert result["complete_answer"] == 1.0


def test_comparison_separates_coverage_from_semantic_judge_score():
    cases = [
        {
            "query_id": "cmp_1",
            "task_type": "comparison",
            "required_doc_ids": ["doc_a", "doc_b"],
            "gold_evidence": [
                {"source_chunk_id": "chunk_a"},
                {"source_chunk_id": "chunk_b"},
            ],
        }
    ]
    predictions = [
        make_prediction(
            "cmp_1",
            cited_doc_ids=["doc_a"],
            cited_chunk_ids=["chunk_a"],
            comparison_score=0.75,
        )
    ]

    result = evaluate_rag_benchmark(cases, predictions)["comparison"]

    assert result["required_doc_coverage"] == 0.5
    assert result["evidence_chunk_recall"] == 0.5
    assert result["semantic_judged_count"] == 1
    assert result["comparison_score"] == pytest.approx(0.75)


def test_unanswerable_measures_action_accuracy_and_wrong_answer_rate():
    cases = [
        {
            "query_id": "na_1",
            "task_type": "unanswerable",
            "expected_action": "abstain",
        },
        {
            "query_id": "na_2",
            "task_type": "unanswerable",
            "expected_action": "external_research",
        },
    ]
    predictions = [
        make_prediction("na_1", predicted_action="abstain"),
        make_prediction("na_2", predicted_action="answer_with_citations"),
    ]

    result = evaluate_rag_benchmark(cases, predictions)["unanswerable"]

    assert result["action_accuracy"] == 0.5
    assert result["unsupported_answer"] == 0.5
    assert result["by_expected_action"]["abstain"]["accuracy"] == 1.0
    assert result["by_expected_action"]["external_research"]["accuracy"] == 0.0


def test_query_ids_must_match_exactly():
    cases = [
        {
            "query_id": "fact_1",
            "task_type": "single_fact",
            "gold_facts": [["答案"]],
        }
    ]

    with pytest.raises(ValueError, match="query_id 不一致"):
        evaluate_rag_benchmark(cases, [make_prediction("another_id")])
