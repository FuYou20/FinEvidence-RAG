import json
from copy import deepcopy

import pytest

from src.embedder import embed_chunks


def make_chunks(count=5):
    return [
        {
            "chunk_id": f"doc:p1:c{index}",
            "doc_id": "doc",
            "page_number": 1,
            "text": "x" * (index + 1),
        }
        for index in range(count)
    ]


def test_embed_chunks_batches_requests_and_preserves_order():
    calls = []

    def fake_embed_batch(texts):
        calls.append(list(texts))
        return [
            [float(len(text)), 1.0, 2.0]
            for text in texts
        ]

    chunks = make_chunks(5)
    results = embed_chunks(
        chunks,
        fake_embed_batch,
        model_name="fake-model",
        batch_size=2,
    )

    assert [len(batch) for batch in calls] == [2, 2, 1]
    assert [item["chunk_id"] for item in results] == [
        item["chunk_id"] for item in chunks
    ]
    assert [item["embedding"][0] for item in results] == [
        float(len(item["text"])) for item in chunks
    ]
    assert all(item["embedding_model"] == "fake-model" for item in results)
    assert all(item["embedding_dimension"] == 3 for item in results)


def test_embed_chunks_rejects_wrong_vector_count():
    def missing_one_vector(texts):
        return [[1.0, 2.0, 3.0] for _ in texts[:-1]]

    with pytest.raises(ValueError, match="数量"):
        embed_chunks(
            make_chunks(3),
            missing_one_vector,
            model_name="fake-model",
            batch_size=3,
        )


def test_embed_chunks_rejects_empty_vector():
    def empty_vectors(texts):
        return [[] for _ in texts]

    with pytest.raises(ValueError, match="维度为0"):
        embed_chunks(
            make_chunks(1),
            empty_vectors,
            model_name="fake-model",
        )


def test_embed_chunks_rejects_inconsistent_dimensions():
    def inconsistent_vectors(texts):
        return [
            [1.0, 2.0, 3.0] if text != "xxx" else [1.0, 2.0]
            for text in texts
        ]

    with pytest.raises(ValueError, match="长度不统一"):
        embed_chunks(
            make_chunks(4),
            inconsistent_vectors,
            model_name="fake-model",
            batch_size=2,
        )


@pytest.mark.parametrize("batch_size", [0, -1, 1.5])
def test_embed_chunks_rejects_invalid_batch_size(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        embed_chunks(
            make_chunks(1),
            lambda texts: [[1.0] for _ in texts],
            model_name="fake-model",
            batch_size=batch_size,
        )


def test_embed_chunks_requires_callable_embed_batch():
    with pytest.raises(TypeError, match="embed_batch"):
        embed_chunks(
            make_chunks(1),
            None,
            model_name="fake-model",
        )


def test_embed_chunks_empty_input_does_not_call_provider():
    def should_not_be_called(_texts):
        raise AssertionError("空输入不应调用 Embedding Provider")

    assert embed_chunks(
        [],
        should_not_be_called,
        model_name="fake-model",
    ) == []


def test_embed_chunks_does_not_mutate_input_and_is_json_serializable():
    chunks = make_chunks(3)
    original = deepcopy(chunks)

    results = embed_chunks(
        chunks,
        lambda texts: [[float(len(text)), 1.0] for text in texts],
        model_name="fake-model",
        batch_size=2,
    )

    assert chunks == original
    assert results is not chunks
    json.dumps(results, ensure_ascii=False)
