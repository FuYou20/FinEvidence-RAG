from copy import deepcopy
from types import SimpleNamespace

import pytest
import requests

from errors import RequestError
from src.reranker import create_dashscope_rerank_batch, rerank_results


class FakeResponse:
    def __init__(self, payload, *, status_code=200, error=None):
        self.payload = payload
        self.status_code = status_code
        self.error = error

    def raise_for_status(self):
        if self.error is not None:
            raise self.error

    def json(self):
        return self.payload


def test_rerank_results_reorders_and_preserves_retrieval_scores():
    candidates = [
        {"chunk_id": "A", "text": "一般背景", "rank": 1, "score": 0.91},
        {"chunk_id": "B", "text": "直接回答", "rank": 2, "score": 0.82},
        {"chunk_id": "C", "text": "无关内容", "rank": 3, "score": 0.75},
    ]
    before = deepcopy(candidates)

    def fake_rerank_batch(query, documents, *, top_n):
        assert query == "公司的收入是多少？"
        assert documents == ["一般背景", "直接回答", "无关内容"]
        assert top_n == 2
        return [
            {"index": 1, "relevance_score": 0.98},
            {"index": 0, "relevance_score": 0.55},
        ]

    results = rerank_results(
        "公司的收入是多少？",
        candidates,
        fake_rerank_batch,
        top_k=2,
    )

    assert [item["chunk_id"] for item in results] == ["B", "A"]
    assert [item["rank"] for item in results] == [1, 2]
    assert results[0]["retrieval_rank"] == 2
    assert results[0]["retrieval_score"] == pytest.approx(0.82)
    assert results[0]["rerank_score"] == pytest.approx(0.98)
    assert results[0]["score"] == pytest.approx(0.98)
    assert candidates == before


def test_rerank_results_rejects_duplicate_provider_indices():
    candidates = [
        {"chunk_id": "A", "text": "a", "rank": 1, "score": 0.9},
        {"chunk_id": "B", "text": "b", "rank": 2, "score": 0.8},
    ]

    with pytest.raises(ValueError, match="下标重复"):
        rerank_results(
            "query",
            candidates,
            lambda *_args, **_kwargs: [
                {"index": 0, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.8},
            ],
            top_k=2,
        )


def test_dashscope_rerank_batch_builds_expected_request():
    calls = []
    response = FakeResponse(
        {
            "output": {
                "results": [
                    {"index": 1, "relevance_score": 0.95},
                    {"index": 0, "relevance_score": 0.42},
                ]
            }
        }
    )

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return response

    rerank_batch = create_dashscope_rerank_batch(
        api_key="secret",
        model_name="qwen3.7-text-rerank",
        http_client=SimpleNamespace(post=post),
    )
    results = rerank_batch("问题", ["文档一", "文档二"], top_n=2)

    request_json = calls[0][1]["json"]
    assert request_json["model"] == "qwen3.7-text-rerank"
    assert request_json["input"] == {
        "query": "问题",
        "documents": ["文档一", "文档二"],
    }
    assert request_json["parameters"]["top_n"] == 2
    assert results[0]["index"] == 1


def test_dashscope_rerank_batch_maps_timeout_to_retryable_error():
    def post(*_args, **_kwargs):
        raise requests.exceptions.Timeout("timeout")

    rerank_batch = create_dashscope_rerank_batch(
        api_key="secret",
        http_client=SimpleNamespace(post=post),
    )

    with pytest.raises(RequestError) as exc_info:
        rerank_batch("问题", ["候选"], top_n=1)

    assert exc_info.value.retryable is True
    assert exc_info.value.status_code == 408


@pytest.mark.parametrize(
    "status_code, expected_retryable",
    [(400, False), (429, True), (503, True)],
)
def test_dashscope_rerank_batch_http_retry_policy(
    status_code,
    expected_retryable,
):
    error_response = SimpleNamespace(status_code=status_code)
    error = requests.exceptions.HTTPError(response=error_response)
    response = FakeResponse({}, status_code=status_code, error=error)
    rerank_batch = create_dashscope_rerank_batch(
        api_key="secret",
        http_client=SimpleNamespace(
            post=lambda *_args, **_kwargs: response
        ),
    )

    with pytest.raises(RequestError) as exc_info:
        rerank_batch("问题", ["候选"], top_n=1)

    assert exc_info.value.status_code == status_code
    assert exc_info.value.retryable is expected_retryable
