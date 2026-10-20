from copy import deepcopy

import pytest

from src.evaluation.error_analysis import (
    analyze_retrieval_errors,
    write_error_analysis_jsonl,
    write_error_analysis_markdown,
)


def make_case(*, query_id="q1", gold_pages=(1, 2)):
    return {
        "query_id": query_id,
        "question": "测试问题",
        "gold_evidence": [
            {
                "doc_id": "doc-a",
                "page_number": page_number,
                "relevance": 2 if index == 0 else 1,
            }
            for index, page_number in enumerate(gold_pages)
        ],
    }


def make_run(*, query_id="q1", retrieved_pages=(1, 2)):
    return {
        "query_id": query_id,
        "question": "测试问题",
        "retrieved_pages": [
            {
                "evidence_id": f"doc-a:p{page_number}",
                "doc_id": "doc-a",
                "page_number": page_number,
                "rank": rank,
                "score": 1.0 - rank / 10,
            }
            for rank, page_number in enumerate(retrieved_pages, start=1)
        ],
    }


def test_analyze_retrieval_errors_reports_complete_recall():
    result = analyze_retrieval_errors(
        [make_case()],
        [make_run()],
    )[0]

    assert result["status"] == "complete"
    assert result["hit_count"] == 2
    assert result["recall"] == 1.0
    assert result["first_relevant_rank"] == 1
    assert result["missed_gold_pages"] == []
    assert result["irrelevant_retrieved_pages"] == []


def test_analyze_retrieval_errors_reports_partial_recall():
    result = analyze_retrieval_errors(
        [make_case()],
        [make_run(retrieved_pages=(9, 2))],
    )[0]

    assert result["status"] == "partial_recall"
    assert result["hit_count"] == 1
    assert result["recall"] == 0.5
    assert result["first_relevant_rank"] == 2
    assert [page["page_number"] for page in result["missed_gold_pages"]] == [1]
    assert [
        page["page_number"]
        for page in result["irrelevant_retrieved_pages"]
    ] == [9]


def test_analyze_retrieval_errors_reports_no_recall():
    result = analyze_retrieval_errors(
        [make_case()],
        [make_run(retrieved_pages=(8, 9))],
    )[0]

    assert result["status"] == "no_recall"
    assert result["hit_count"] == 0
    assert result["recall"] == 0.0
    assert result["first_relevant_rank"] is None


def test_same_document_different_page_is_not_a_hit():
    result = analyze_retrieval_errors(
        [make_case(gold_pages=(1,))],
        [make_run(retrieved_pages=(2,))],
    )[0]

    assert result["status"] == "no_recall"


def test_duplicate_gold_quotes_on_same_page_are_merged():
    case = make_case(gold_pages=(1,))
    case["gold_evidence"].append(
        {
            "doc_id": "doc-a",
            "page_number": 1,
            "relevance": 1,
        }
    )

    result = analyze_retrieval_errors(
        [case],
        [make_run(retrieved_pages=(1,))],
    )[0]

    assert result["gold_page_count"] == 1
    assert result["recall"] == 1.0
    assert result["gold_pages"][0]["relevance"] == 2
    assert result["gold_pages"][0]["evidence_count"] == 2


def test_analyze_retrieval_errors_rejects_mismatched_query_ids():
    with pytest.raises(ValueError, match="query_id 不一致"):
        analyze_retrieval_errors(
            [make_case(query_id="q1")],
            [make_run(query_id="q2")],
        )


def test_analyze_retrieval_errors_does_not_mutate_inputs():
    cases = [make_case()]
    runs = [make_run(retrieved_pages=(9, 2))]
    cases_before = deepcopy(cases)
    runs_before = deepcopy(runs)

    analyze_retrieval_errors(cases, runs)

    assert cases == cases_before
    assert runs == runs_before


def test_error_analysis_writers_create_machine_and_human_reports(tmp_path):
    analyses = analyze_retrieval_errors(
        [make_case()],
        [make_run(retrieved_pages=(9, 2))],
    )
    jsonl_path = tmp_path / "analysis.jsonl"
    markdown_path = tmp_path / "analysis.md"

    write_error_analysis_jsonl(jsonl_path, analyses)
    write_error_analysis_markdown(markdown_path, analyses)

    assert '"query_id": "q1"' in jsonl_path.read_text(encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Retrieval Error Analysis" in markdown
    assert "doc-a:p1" in markdown
