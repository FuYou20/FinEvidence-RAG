import math

import pytest

from src.evaluation.table_retrieval_metrics import (
    evaluate_table_retrieval,
    load_table_retrieval_cases,
)


def make_case():
    return {
        "query_id": "table_001",
        "question": "目标值是多少？",
        "answer_values": ["40.78", "分钟"],
        "gold_evidence": [
            {
                "doc_id": "doc",
                "page_number": 2,
                "table_id": "doc:p2:t0",
                "content_type": "table",
                "relevance": 2,
            },
            {
                "doc_id": "doc",
                "page_number": 3,
                "table_id": "doc:p3:t0",
                "content_type": "table",
                "relevance": 1,
            },
        ],
    }


def test_table_metrics_use_overall_rank_and_match_answer_values():
    run_records = [
        {
            "query_id": "table_001",
            "retrieved_chunks": [
                {
                    "content_type": "text",
                    "table_id": None,
                    "text": "同一页正文不应算表格命中",
                },
                {
                    "content_type": "table",
                    "table_id": "noise:p9:t0",
                    "text": "无关表格",
                },
                {
                    "content_type": "table",
                    "table_id": "doc:p2:t0",
                    "text": "Boomplay | 40.78 分钟",
                },
                {
                    "content_type": "table",
                    "table_id": "doc:p2:t0",
                    "text": "同一表格的另一个 chunk",
                },
            ],
        }
    ]

    report = evaluate_table_retrieval(
        [make_case()],
        run_records,
        top_k=4,
    )

    expected_dcg = 3.0 / math.log2(4)
    ideal_dcg = 3.0 + 1.0 / math.log2(3)
    assert report["table_success_at_k"] == 1.0
    assert report["table_mrr_at_k"] == pytest.approx(1 / 3)
    assert report["table_recall_at_k"] == pytest.approx(0.5)
    assert report["table_ndcg_at_k"] == pytest.approx(
        expected_dcg / ideal_dcg
    )
    assert report["answer_value_hit_at_k"] == 1.0
    assert report["per_query"][0]["first_relevant_rank"] == 3


def test_same_page_text_does_not_count_as_table_hit():
    run_records = [
        {
            "query_id": "table_001",
            "retrieved_chunks": [
                {
                    "doc_id": "doc",
                    "page_number": 2,
                    "content_type": "text",
                    "table_id": None,
                    "text": "40.78分钟",
                }
            ],
        }
    ]

    report = evaluate_table_retrieval(
        [make_case()],
        run_records,
        top_k=1,
    )

    assert report["table_success_at_k"] == 0.0
    assert report["answer_value_hit_at_k"] == 0.0


def test_answer_value_hit_requires_all_expected_values():
    run_records = [
        {
            "query_id": "table_001",
            "retrieved_chunks": [
                {
                    "content_type": "table",
                    "table_id": "doc:p2:t0",
                    "text": "Boomplay | 40.78",
                }
            ],
        }
    ]

    report = evaluate_table_retrieval(
        [make_case()],
        run_records,
        top_k=1,
    )

    assert report["table_success_at_k"] == 1.0
    assert report["answer_value_hit_at_k"] == 0.0
    assert report["answer_value_recall_at_k"] == pytest.approx(0.5)


def test_load_table_cases_validates_table_specific_fields(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        '{"query_id":"q1","question":"问题",'
        '"answer_values":["42"],"gold_evidence":['
        '{"doc_id":"doc","page_number":1,"relevance":2,'
        '"table_id":"doc:p1:t0","content_type":"table"}]}\n',
        encoding="utf-8",
    )

    cases = load_table_retrieval_cases(cases_path)

    assert cases[0]["answer_values"] == ["42"]


@pytest.mark.parametrize(
    "field_fragment",
    [
        '"answer_values":[]',
        '"answer_values":["42"]',
    ],
)
def test_load_table_cases_rejects_invalid_protocol(tmp_path, field_fragment):
    evidence = (
        '{"doc_id":"doc","page_number":1,"relevance":2,'
        '"content_type":"table"}'
        if field_fragment == '"answer_values":["42"]'
        else (
            '{"doc_id":"doc","page_number":1,"relevance":2,'
            '"table_id":"doc:p1:t0","content_type":"table"}'
        )
    )
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        "{" + '"query_id":"q1","question":"问题",' +
        field_fragment + ',"gold_evidence":[' + evidence + "]}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_table_retrieval_cases(cases_path)
