import math

import pytest

from src.evaluation.retrieval_metrics import (
    evaluate_chunk_retrieval,
    evaluate_retrieval,
)


def test_evaluate_retrieval_computes_page_level_metrics():
    cases = [
        {
            "query_id": "q1",
            "question": "问题一",
            "gold_evidence": [
                {"doc_id": "doc_a", "page_number": 1, "relevance": 1},
                # 同页第二条 quote 的相关度更高，评测时应取 max=2。
                {"doc_id": "doc_a", "page_number": 1, "relevance": 2},
                {"doc_id": "doc_b", "page_number": 2, "relevance": 1},
            ],
        },
        {
            "query_id": "q2",
            "question": "问题二",
            "gold_evidence": [
                {"doc_id": "doc_c", "page_number": 3, "relevance": 2}
            ],
        },
    ]
    run_records = [
        {
            "query_id": "q1",
            "retrieved_pages": [
                {"evidence_id": "doc_b:p2"},
                {"evidence_id": "noise:p9"},
                {"evidence_id": "doc_a:p1"},
            ],
        },
        {
            "query_id": "q2",
            "retrieved_pages": [
                {"evidence_id": "noise:p8"},
            ],
        },
    ]

    report = evaluate_retrieval(cases, run_records, top_k=3)

    q1_dcg = 1.0 + 3.0 / math.log2(4)
    q1_ideal_dcg = 3.0 + 1.0 / math.log2(3)
    q1_ndcg = q1_dcg / q1_ideal_dcg

    assert report["query_count"] == 2
    assert report["success_at_k"] == pytest.approx(0.5)
    assert report["mrr_at_k"] == pytest.approx(0.5)
    assert report["recall_at_k"] == pytest.approx(0.5)
    assert report["ndcg_at_k"] == pytest.approx(q1_ndcg / 2)

    q1 = report["per_query"][0]
    assert q1["relevant_page_count"] == 2
    assert q1["retrieved_relevant_page_count"] == 2
    assert q1["first_relevant_rank"] == 1
    assert q1["recall_at_k"] == pytest.approx(1.0)


def test_evaluate_retrieval_rejects_missing_run_record():
    cases = [
        {
            "query_id": "q1",
            "gold_evidence": [
                {"doc_id": "doc_a", "page_number": 1, "relevance": 2}
            ],
        }
    ]

    with pytest.raises(ValueError, match="run_records 缺少 query_id"):
        evaluate_retrieval(
            cases,
            [{"query_id": "another", "retrieved_pages": []}],
            top_k=5,
        )


def test_chunk_metrics_distinguish_two_gold_chunks_on_same_page():
    cases = [
        {
            "query_id": "q1",
            "question": "跨 chunk 问题",
            "gold_evidence": [
                {
                    "doc_id": "doc_a",
                    "page_number": 1,
                    "source_chunk_id": "doc_a:p1:c0",
                    "relevance": 3,
                },
                {
                    "doc_id": "doc_a",
                    "page_number": 1,
                    "source_chunk_id": "doc_a:p1:c1",
                    "relevance": 2,
                },
            ],
        }
    ]
    run_records = [
        {
            "query_id": "q1",
            "retrieved_pages": [{"evidence_id": "doc_a:p1"}],
            "retrieved_chunks": [
                {"chunk_id": "doc_a:p1:c0"},
                {"chunk_id": "noise:p9:c0"},
            ],
        }
    ]

    page_report = evaluate_retrieval(cases, run_records, top_k=2)
    chunk_report = evaluate_chunk_retrieval(cases, run_records, top_k=2)

    assert page_report["recall_at_k"] == 1.0
    assert chunk_report["success_at_k"] == 1.0
    assert chunk_report["recall_at_k"] == 0.5
    assert chunk_report["per_query"][0]["relevant_chunk_count"] == 2


def test_chunk_metrics_require_source_chunk_id():
    with pytest.raises(ValueError, match="source_chunk_id"):
        evaluate_chunk_retrieval(
            [
                {
                    "query_id": "q1",
                    "gold_evidence": [
                        {
                            "doc_id": "doc_a",
                            "page_number": 1,
                            "relevance": 1,
                        }
                    ],
                }
            ],
            [{"query_id": "q1", "retrieved_chunks": []}],
            top_k=5,
        )

