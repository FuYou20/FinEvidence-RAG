from copy import deepcopy

import pytest

from src.chunk_expander import expand_neighbor_chunks


@pytest.fixture
def metadata():
    return [
        {
            "chunk_id": "doc_a:p1:c0",
            "doc_id": "doc_a",
            "page_number": 1,
            "document_chunk_index": 0,
            "text": "a0",
        },
        {
            "chunk_id": "doc_a:p1:c1",
            "doc_id": "doc_a",
            "page_number": 1,
            "document_chunk_index": 1,
            "text": "a1",
        },
        {
            "chunk_id": "doc_a:p2:c0",
            "doc_id": "doc_a",
            "page_number": 2,
            "document_chunk_index": 2,
            "text": "a2",
        },
        {
            "chunk_id": "doc_a:p2:c1",
            "doc_id": "doc_a",
            "page_number": 2,
            "document_chunk_index": 3,
            "text": "a3",
        },
        {
            "chunk_id": "doc_b:p1:c0",
            "doc_id": "doc_b",
            "page_number": 1,
            "document_chunk_index": 0,
            "text": "b0",
        },
    ]


def test_expands_across_page_boundary_with_decayed_scores(metadata):
    results = [
        {
            **metadata[1],
            "rank": 1,
            "score": 1.0,
        }
    ]

    expanded = expand_neighbor_chunks(
        results,
        metadata,
        top_k=3,
        window=1,
        decay=0.8,
    )

    assert [item["chunk_id"] for item in expanded] == [
        "doc_a:p1:c1",
        "doc_a:p1:c0",
        "doc_a:p2:c0",
    ]
    assert expanded[1]["score"] == pytest.approx(0.8)
    assert expanded[2]["score"] == pytest.approx(0.8)
    assert expanded[2]["expanded_from"] == "doc_a:p1:c1"
    assert expanded[2]["neighbor_distance"] == 1


def test_never_crosses_document_boundary(metadata):
    results = [{**metadata[3], "rank": 1, "score": 1.0}]

    expanded = expand_neighbor_chunks(
        results,
        metadata,
        top_k=5,
        window=2,
    )

    assert all(item["doc_id"] == "doc_a" for item in expanded)


def test_multiple_seeds_keep_highest_neighbor_score(metadata):
    results = [
        {**metadata[0], "rank": 1, "score": 1.0},
        {**metadata[2], "rank": 2, "score": 0.9},
    ]

    expanded = expand_neighbor_chunks(
        results,
        metadata,
        top_k=4,
        seed_top_n=2,
        window=1,
        decay=0.8,
    )
    by_id = {item["chunk_id"]: item for item in expanded}

    assert by_id["doc_a:p1:c1"]["score"] == pytest.approx(0.8)
    assert by_id["doc_a:p1:c1"]["expanded_from"] == "doc_a:p1:c0"


def test_direct_hit_can_receive_stronger_neighbor_support(metadata):
    results = [
        {**metadata[0], "rank": 1, "score": 1.0},
        {**metadata[1], "rank": 2, "score": 0.5},
    ]

    expanded = expand_neighbor_chunks(
        results,
        metadata,
        top_k=3,
        seed_top_n=1,
        decay=0.8,
    )
    by_id = {item["chunk_id"]: item for item in expanded}

    assert by_id["doc_a:p1:c1"]["retrieval_score"] == pytest.approx(0.5)
    assert by_id["doc_a:p1:c1"]["neighbor_score"] == pytest.approx(0.8)
    assert by_id["doc_a:p1:c1"]["score"] == pytest.approx(0.8)
    assert by_id["doc_a:p1:c1"]["is_direct_hit"] is True


def test_old_metadata_order_is_compatible_without_explicit_position(metadata):
    old_metadata = [
        {key: value for key, value in item.items() if key != "document_chunk_index"}
        for item in metadata
    ]
    results = [{**old_metadata[1], "rank": 1, "score": 1.0}]

    expanded = expand_neighbor_chunks(
        results,
        old_metadata,
        top_k=3,
        window=1,
    )

    assert {item["chunk_id"] for item in expanded} == {
        "doc_a:p1:c0",
        "doc_a:p1:c1",
        "doc_a:p2:c0",
    }


def test_inputs_are_not_mutated(metadata):
    results = [{**metadata[1], "rank": 1, "score": 1.0}]
    original_metadata = deepcopy(metadata)
    original_results = deepcopy(results)

    expand_neighbor_chunks(results, metadata, top_k=3)

    assert metadata == original_metadata
    assert results == original_results


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("top_k", 0),
        ("seed_top_n", 0),
        ("window", 0),
        ("decay", 0),
        ("decay", 1.1),
    ],
)
def test_rejects_invalid_parameters(metadata, argument, value):
    kwargs = {argument: value}
    with pytest.raises(ValueError):
        expand_neighbor_chunks(
            [{**metadata[0], "rank": 1, "score": 1.0}],
            metadata,
            **kwargs,
        )
