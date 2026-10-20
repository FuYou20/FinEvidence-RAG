from types import SimpleNamespace

import pytest

import src.embedder as embedder


class FakeEmbeddings:
    """记录请求，并返回预先准备好的向量。"""

    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=vector)
                for vector in self.vectors
            ]
        )


def make_fake_client(vectors):
    return SimpleNamespace(embeddings=FakeEmbeddings(vectors))


def test_qwen_embed_batch_sends_one_batch_request(monkeypatch):
    fake_client = make_fake_client(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    monkeypatch.setattr(embedder, "qwen_embed_client", fake_client)

    texts = ["第一段", "第二段", "第三段"]
    vectors = embedder.qwen_embed_batch(
        texts,
        model_name="test-embedding-model",
    )

    assert fake_client.embeddings.calls == [
        {
            "model": "test-embedding-model",
            "input": texts,
        }
    ]
    assert vectors == [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ]


def test_qwen_embed_batch_rejects_wrong_vector_count(monkeypatch):
    fake_client = make_fake_client([[1.0, 2.0]])
    monkeypatch.setattr(embedder, "qwen_embed_client", fake_client)

    with pytest.raises(ValueError, match="向量数目"):
        embedder.qwen_embed_batch(["第一段", "第二段"])


def test_qwen_embed_batch_empty_input_skips_api(monkeypatch):
    class MustNotCallEmbeddings:
        def create(self, **_kwargs):
            raise AssertionError("空输入不应请求真实 Embedding API")

    fake_client = SimpleNamespace(embeddings=MustNotCallEmbeddings())
    monkeypatch.setattr(embedder, "qwen_embed_client", fake_client)

    assert embedder.qwen_embed_batch([]) == []
