from copy import deepcopy

import pytest

import src.hybrid_retriever as hybrid_module
from src.hybrid_retriever import reciprocal_rank_fusion


@pytest.fixture
def result_lists():
    dense = [
        {"chunk_id": "A", "text": "both", "rank": 1, "score": 0.9},
        {"chunk_id": "B", "text": "dense-only", "rank": 2, "score": 0.8},
    ]
    sparse = [
        {"chunk_id": "C", "text": "sparse-only", "rank": 1, "score": 9.0},
        {"chunk_id": "A", "text": "both", "rank": 2, "score": 8.0},
    ]
    return dense, sparse


def test_rrf_fuses_shared_and_single_source_results(result_lists):
    dense, sparse = result_lists

    results = reciprocal_rank_fusion(dense, sparse, top_k=3)
    by_id = {item["chunk_id"]: item for item in results}

    assert [item["chunk_id"] for item in results] == ["A", "C", "B"]
    assert by_id["A"]["retrieval_source"] == ["dense", "bm25"]
    assert by_id["B"]["retrieval_source"] == ["dense"]
    assert by_id["C"]["retrieval_source"] == ["bm25"]
    assert by_id["B"]["sparse_rank"] is None
    assert by_id["C"]["dense_rank"] is None


def test_rrf_preserves_source_scores_and_rebuilds_common_score(result_lists):
    dense, sparse = result_lists

    results = reciprocal_rank_fusion(dense, sparse, top_k=3)
    first = results[0]

    assert first["chunk_id"] == "A"
    assert first["dense_score"] == pytest.approx(0.9)
    assert first["sparse_score"] == pytest.approx(8.0)
    assert first["score"] == pytest.approx(first["rrf_score"])
    assert [item["rank"] for item in results] == [1, 2, 3]


def test_rrf_uses_weights_and_rrf_k():
    dense = [{"chunk_id": "A", "rank": 1, "score": 0.9}]
    sparse = [{"chunk_id": "A", "rank": 2, "score": 9.0}]

    result = reciprocal_rank_fusion(
        dense,
        sparse,
        rrf_k=10,
        dense_weight=2.0,
        sparse_weight=3.0,
    )[0]

    assert result["rrf_score"] == pytest.approx(2 / 11 + 3 / 12)


def test_rrf_honors_top_k(result_lists):
    dense, sparse = result_lists

    results = reciprocal_rank_fusion(dense, sparse, top_k=2)

    assert len(results) == 2


def test_rrf_does_not_mutate_inputs(result_lists):
    dense, sparse = result_lists
    dense_before = deepcopy(dense)
    sparse_before = deepcopy(sparse)

    reciprocal_rank_fusion(dense, sparse)

    assert dense == dense_before
    assert sparse == sparse_before


def test_rrf_allows_one_empty_source():
    dense = [{"chunk_id": "A", "rank": 1, "score": 0.9}]

    result = reciprocal_rank_fusion(dense, [])

    assert result[0]["retrieval_source"] == ["dense"]
    assert result[0]["sparse_rank"] is None


@pytest.mark.parametrize("name", ["dense", "sparse"])
def test_rrf_rejects_duplicate_chunk_ids(name):
    duplicate = [
        {"chunk_id": "A", "rank": 1, "score": 1.0},
        {"chunk_id": "A", "rank": 2, "score": 0.5},
    ]
    dense = duplicate if name == "dense" else []
    sparse = duplicate if name == "sparse" else []

    with pytest.raises(ValueError, match="chunk_id 重复"):
        reciprocal_rank_fusion(dense, sparse)


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_rrf_rejects_invalid_top_k(top_k):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], [], top_k=top_k)


def test_hybrid_retrieve_calls_both_retrievers_and_fuses(monkeypatch):
    calls = []

    def fake_dense_retrieve(**kwargs):
        calls.append(("dense", kwargs["top_k"]))
        return [{"chunk_id": "A", "rank": 1, "score": 0.8}]

    def fake_sparse_retrieve(**kwargs):
        calls.append(("sparse", kwargs["top_k"]))
        return [{"chunk_id": "A", "rank": 1, "score": 8.0}]

    monkeypatch.setattr(hybrid_module, "dense_retrieve", fake_dense_retrieve)
    monkeypatch.setattr(hybrid_module, "sparse_retrieve", fake_sparse_retrieve)

    results = hybrid_module.hybrid_retrieve(
        "query",
        dense_index=object(),
        bm25_index=object(),
        metadata=[{"chunk_id": "A"}],
        embed_batch=lambda texts: texts,
        dense_k=12,
        sparse_k=15,
        top_k=5,
    )

    assert calls == [("dense", 12), ("sparse", 15)]
    assert results[0]["retrieval_source"] == ["dense", "bm25"]
