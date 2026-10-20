from copy import deepcopy
import json

import numpy as np
import pytest

from src.vector_store import build_index, search_index


def make_embedded_chunks():
    return [
        {
            "chunk_id": "doc:p1:c0",
            "doc_id": "doc",
            "source_path": "/fake/report.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "char_start": 0,
            "char_end": 4,
            "text": "苹果公司",
            "chunking_version": "char_4_overlap_0",
            "embedding": [3.0, 4.0],
            "embedding_model": "fake-model",
            "embedding_dimension": 2,
        },
        {
            "chunk_id": "doc:p2:c0",
            "doc_id": "doc",
            "source_path": "/fake/report.pdf",
            "page_number": 2,
            "chunk_index": 0,
            "char_start": 0,
            "char_end": 4,
            "text": "银行利率",
            "chunking_version": "char_4_overlap_0",
            "embedding": [0.0, 2.0],
            "embedding_model": "fake-model",
            "embedding_dimension": 2,
        },
    ]


def make_search_chunks():
    return [
        {
            "chunk_id": "doc:p1:c0",
            "doc_id": "doc",
            "source_path": "/fake/report.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "char_start": 0,
            "char_end": 4,
            "text": "苹果公司",
            "chunking_version": "char_4_overlap_0",
            "embedding": [1.0, 0.0],
            "embedding_model": "fake-model",
            "embedding_dimension": 2,
        },
        {
            "chunk_id": "doc:p2:c0",
            "doc_id": "doc",
            "source_path": "/fake/report.pdf",
            "page_number": 2,
            "chunk_index": 0,
            "char_start": 0,
            "char_end": 4,
            "text": "苹果手机",
            "chunking_version": "char_4_overlap_0",
            "embedding": [0.8, 0.2],
            "embedding_model": "fake-model",
            "embedding_dimension": 2,
        },
        {
            "chunk_id": "doc:p3:c0",
            "doc_id": "doc",
            "source_path": "/fake/report.pdf",
            "page_number": 3,
            "chunk_index": 0,
            "char_start": 0,
            "char_end": 4,
            "text": "银行利率",
            "chunking_version": "char_4_overlap_0",
            "embedding": [0.0, 1.0],
            "embedding_model": "fake-model",
            "embedding_dimension": 2,
        },
    ]


def test_build_index_adds_all_vectors_and_preserves_metadata_order():
    chunks = make_embedded_chunks()

    index, metadata = build_index(chunks)

    assert index.ntotal == 2
    assert index.d == 2
    assert [item["chunk_id"] for item in metadata] == [
        "doc:p1:c0",
        "doc:p2:c0",
    ]
    assert metadata[0]["doc_id"] == "doc"
    assert metadata[0]["source_path"] == "/fake/report.pdf"
    assert metadata[0]["char_start"] == 0
    assert metadata[0]["char_end"] == 4
    assert metadata[0]["chunking_version"] == "char_4_overlap_0"
    assert metadata[0]["embedding_model"] == "fake-model"
    assert metadata[0]["embedding_dimension"] == 2
    assert all("embedding" not in item for item in metadata)


def test_build_index_normalizes_vectors_to_unit_length():
    index, _metadata = build_index(make_embedded_chunks())

    stored_vectors = np.vstack(
        [index.reconstruct(position) for position in range(index.ntotal)]
    )
    norms = np.linalg.norm(stored_vectors, axis=1)

    np.testing.assert_allclose(norms, np.ones(2), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("chunks", [None, []])
def test_build_index_rejects_empty_input(chunks):
    with pytest.raises(ValueError):
        build_index(chunks)


def test_build_index_rejects_zero_vector():
    chunks = make_embedded_chunks()
    chunks[1]["embedding"] = [0.0, 0.0]

    with pytest.raises(ValueError, match="norm|零|zero"):
        build_index(chunks)


def test_build_index_rejects_inconsistent_dimensions():
    chunks = make_embedded_chunks()
    chunks[1]["embedding"] = [1.0, 2.0, 3.0]

    with pytest.raises(ValueError):
        build_index(chunks)


def test_build_index_does_not_mutate_input():
    chunks = make_embedded_chunks()
    original = deepcopy(chunks)

    build_index(chunks)

    assert chunks == original


def test_search_index_returns_ranked_results_for_single_query():
    index, metadata = build_index(make_search_chunks())

    results = search_index(index, metadata, [1.0, 0.0], top_k=3)

    assert [item["chunk_id"] for item in results] == [
        "doc:p1:c0",
        "doc:p2:c0",
        "doc:p3:c0",
    ]
    assert [item["rank"] for item in results] == [1, 2, 3]
    scores = [item["score"] for item in results]
    assert scores == sorted(scores, reverse=True)


def test_search_index_preserves_source_metadata_without_embedding():
    index, metadata = build_index(make_search_chunks())

    results = search_index(index, metadata, [1.0, 0.0], top_k=1)

    assert len(results) == 1
    result = results[0]
    assert result["doc_id"] == "doc"
    assert result["source_path"] == "/fake/report.pdf"
    assert result["chunk_index"] == 0
    assert result["char_start"] == 0
    assert result["char_end"] == 4
    assert result["chunking_version"] == "char_4_overlap_0"
    assert result["embedding_model"] == "fake-model"
    assert result["embedding_dimension"] == 2
    assert "embedding" not in result
    assert result["index_id"] == 0
    assert result["rank"] == 1
    assert isinstance(result["score"], float)


def test_search_index_clamps_top_k_to_index_size():
    index, metadata = build_index(make_search_chunks())

    results = search_index(index, metadata, [1.0, 0.0], top_k=10)

    assert len(results) == index.ntotal == 3


@pytest.mark.parametrize("top_k", [0, -1, 1.5, "2", None])
def test_search_index_rejects_invalid_top_k(top_k):
    index, metadata = build_index(make_search_chunks())

    with pytest.raises((TypeError, ValueError)):
        search_index(index, metadata, [1.0, 0.0], top_k=top_k)


def test_search_index_rejects_wrong_query_dimension():
    index, metadata = build_index(make_search_chunks())

    with pytest.raises(ValueError):
        search_index(index, metadata, [1.0, 0.0, 0.0])


def test_search_index_rejects_zero_query_vector():
    index, metadata = build_index(make_search_chunks())

    with pytest.raises(ValueError, match="零|zero|norm"):
        search_index(index, metadata, [0.0, 0.0])


def test_search_index_rejects_metadata_size_mismatch():
    index, metadata = build_index(make_search_chunks())

    with pytest.raises(ValueError):
        search_index(index, metadata[:-1], [1.0, 0.0])


def test_search_index_does_not_mutate_inputs_and_is_json_serializable():
    index, metadata = build_index(make_search_chunks())
    query = np.array([3.0, 4.0], dtype=np.float32)
    original_query = query.copy()
    original_metadata = deepcopy(metadata)

    results = search_index(index, metadata, query, top_k=2)

    np.testing.assert_array_equal(query, original_query)
    assert metadata == original_metadata
    json.dumps(results, ensure_ascii=False)


def test_search_index_returns_only_top_k_in_similarity_order():
    # 故意让 metadata 原始顺序与相似度顺序不同，防止测试假通过。
    chunks = [
        {
            "chunk_id": "least-similar",
            "page_number": 3,
            "text": "银行利率",
            "embedding": [0.0, 1.0],
        },
        {
            "chunk_id": "most-similar",
            "page_number": 1,
            "text": "苹果公司",
            "embedding": [1.0, 0.0],
        },
        {
            "chunk_id": "second-similar",
            "page_number": 2,
            "text": "苹果手机",
            "embedding": [0.8, 0.2],
        },
    ]
    index, metadata = build_index(chunks)

    results = search_index(index, metadata, [1.0, 0.0], top_k=2)

    assert len(results) == 2
    assert [item["chunk_id"] for item in results] == [
        "most-similar",
        "second-similar",
    ]
    assert [item["rank"] for item in results] == [1, 2]
    assert results[0]["score"] > results[1]["score"]
