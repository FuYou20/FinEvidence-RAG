from copy import deepcopy

import pytest

from src.evaluation import retrieval_runner


def test_collapse_chunks_to_pages_deduplicates_and_reranks():
    chunk_results = [
        {
            "doc_id": "doc_a",
            "page_number": 5,
            "chunk_id": "doc_a:p5:c1",
            "score": 0.91,
        },
        {
            "doc_id": "doc_a",
            "page_number": 5,
            "chunk_id": "doc_a:p5:c0",
            "score": 0.88,
        },
        {
            "doc_id": "doc_b",
            "page_number": 8,
            "chunk_id": "doc_b:p8:c0",
            "score": 0.80,
        },
    ]
    original_results = deepcopy(chunk_results)

    pages = retrieval_runner.collapse_chunks_to_pages(
        chunk_results,
        top_k=5,
    )

    assert chunk_results == original_results
    assert [page["evidence_id"] for page in pages] == [
        "doc_a:p5",
        "doc_b:p8",
    ]
    assert [page["rank"] for page in pages] == [1, 2]
    assert pages[0]["best_chunk_id"] == "doc_a:p5:c1"
    assert pages[0]["best_content_type"] == "text"


def test_run_retrieval_cases_uses_candidate_k_and_excludes_gold(
    monkeypatch,
):
    cases = [
        {
            "query_id": "ret_001",
            "question": "测试问题一",
            "gold_evidence": [
                {"doc_id": "doc_a", "page_number": 1, "relevance": 2}
            ],
        },
        {
            "query_id": "ret_002",
            "question": "测试问题二",
            "gold_evidence": [
                {"doc_id": "doc_b", "page_number": 2, "relevance": 2}
            ],
        },
    ]
    calls = []

    def fake_retrieve(*, query, index, metadata, embed_batch, top_k):
        calls.append((query, top_k))
        return [
            {
                "doc_id": "doc_a",
                "page_number": 1,
                "chunk_id": "doc_a:p1:c0",
                "score": 0.9,
            },
            {
                "doc_id": "doc_a",
                "page_number": 1,
                "chunk_id": "doc_a:p1:c1",
                "score": 0.8,
            },
            {
                "doc_id": "doc_b",
                "page_number": 2,
                "chunk_id": "doc_b:p2:c0",
                "score": 0.7,
            },
        ]

    monkeypatch.setattr(retrieval_runner, "retrieve", fake_retrieve)

    records = retrieval_runner.run_retrieval_cases(
        cases=cases,
        index=object(),
        metadata=[],
        embed_batch=lambda texts: [],
        candidate_k=10,
        page_top_k=2,
    )

    assert calls == [("测试问题一", 10), ("测试问题二", 10)]
    assert len(records) == 2
    assert records[0]["query_id"] == "ret_001"
    assert [page["evidence_id"] for page in records[0]["retrieved_pages"]] == [
        "doc_a:p1",
        "doc_b:p2",
    ]
    assert "gold_evidence" not in records[0]
    assert records[0]["retrieved_chunk_count"] == 3


def test_run_retrieval_cases_rejects_candidate_k_below_page_top_k():
    with pytest.raises(ValueError, match="candidate_k 不能小于 page_top_k"):
        retrieval_runner.run_retrieval_cases(
            cases=[{"query_id": "q1", "question": "问题"}],
            index=object(),
            metadata=[],
            embed_batch=lambda texts: [],
            candidate_k=2,
            page_top_k=3,
        )


def test_generic_runner_uses_injected_retriever():
    cases = [
        {
            "query_id": "q1",
            "question": "关键词问题",
            "gold_evidence": [],
        }
    ]
    calls = []

    def retrieve_fn(*, query, top_k):
        calls.append((query, top_k))
        return [
            {
                "doc_id": "doc",
                "page_number": 3,
                "chunk_id": "doc:p3:c0",
                "score": 1.5,
            }
        ]

    records = retrieval_runner.run_retrieval_cases_with_retriever(
        cases,
        retrieve_fn,
        candidate_k=8,
        page_top_k=3,
    )

    assert calls == [("关键词问题", 8)]
    assert records[0]["retrieved_pages"][0]["evidence_id"] == "doc:p3"
    assert records[0]["expanded_neighbor_count"] == 0
    assert records[0]["boosted_direct_count"] == 0
    assert records[0]["retrieved_chunks"] == [
        {
            "chunk_id": "doc:p3:c0",
            "doc_id": "doc",
            "page_number": 3,
            "rank": 1,
            "score": 1.5,
            "content_type": "text",
            "table_id": None,
            "text": "",
            "is_direct_hit": True,
            "expanded_from": None,
            "neighbor_distance": 0,
        }
    ]


def test_generic_runner_records_neighbor_expansion_diagnostics():
    cases = [{"query_id": "q1", "question": "问题", "gold_evidence": []}]

    def retrieve_fn(**_kwargs):
        return [
            {
                "doc_id": "doc",
                "page_number": 2,
                "chunk_id": "doc:p2:c0",
                "rank": 1,
                "score": 0.9,
                "is_direct_hit": True,
                "neighbor_score": 0.8,
                "expanded_from": "doc:p1:c1",
                "neighbor_distance": 1,
            },
            {
                "doc_id": "doc",
                "page_number": 1,
                "chunk_id": "doc:p1:c1",
                "rank": 2,
                "score": 0.8,
                "is_direct_hit": False,
                "neighbor_score": 0.8,
                "expanded_from": "doc:p2:c0",
                "neighbor_distance": 1,
            },
        ]

    records = retrieval_runner.run_retrieval_cases_with_retriever(
        cases,
        retrieve_fn,
        candidate_k=2,
        page_top_k=2,
    )

    assert records[0]["expanded_neighbor_count"] == 1
    assert records[0]["boosted_direct_count"] == 1
    assert records[0]["retrieved_chunks"][1]["expanded_from"] == "doc:p2:c0"


def test_generic_runner_records_rerank_diagnostics_when_present():
    cases = [{"query_id": "q1", "question": "问题", "gold_evidence": []}]

    def retrieve_fn(**_kwargs):
        return [
            {
                "doc_id": "doc",
                "page_number": 2,
                "chunk_id": "doc:p2:c0",
                "rank": 1,
                "score": 0.97,
                "retrieval_rank": 4,
                "retrieval_score": 0.61,
                "rerank_score": 0.97,
            }
        ]

    records = retrieval_runner.run_retrieval_cases_with_retriever(
        cases,
        retrieve_fn,
        candidate_k=5,
        page_top_k=5,
    )

    chunk = records[0]["retrieved_chunks"][0]
    assert chunk["retrieval_rank"] == 4
    assert chunk["retrieval_score"] == pytest.approx(0.61)
    assert chunk["rerank_score"] == pytest.approx(0.97)


def test_generic_runner_rejects_non_callable_retriever():
    with pytest.raises(TypeError, match="retrieve_fn"):
        retrieval_runner.run_retrieval_cases_with_retriever(
            [{"query_id": "q1", "question": "问题"}],
            None,
        )

