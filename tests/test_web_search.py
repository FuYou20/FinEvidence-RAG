from types import SimpleNamespace

import pytest
import requests

from errors import RequestError
from src.web_search import create_bocha_web_search


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


def test_bocha_search_normalizes_results_for_evidence_pipeline():
    calls = []
    response = FakeResponse(
        {
            "data": {
                "webPages": {
                    "value": [
                        {
                            "name": "行业数据",
                            "url": "https://example.com/report",
                            "snippet": "2026年市场份额数据",
                            "datePublished": "2026-10-01",
                        }
                    ]
                }
            }
        }
    )

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return response

    search = create_bocha_web_search(
        api_key="secret",
        count=3,
        http_client=SimpleNamespace(post=post),
    )
    results = search(question="问题", query="传音 市场份额", evidence=[])

    assert results[0]["doc_id"] == "web:example.com"
    assert results[0]["content_type"] == "text"
    assert results[0]["url"] == "https://example.com/report"
    assert results[0]["published_at"] == "2026-10-01"
    assert calls[0][1]["json"]["query"] == "传音 市场份额"
    assert calls[0][1]["json"]["summary"] is True
    assert calls[0][1]["json"]["count"] == 3


def test_bocha_search_maps_timeout_to_retryable_error():
    def post(*_args, **_kwargs):
        raise requests.exceptions.Timeout("timeout")

    search = create_bocha_web_search(
        api_key="secret",
        http_client=SimpleNamespace(post=post),
    )

    with pytest.raises(RequestError) as exc_info:
        search(question="问题", query="查询", evidence=[])

    assert exc_info.value.retryable is True
    assert exc_info.value.status_code == 408


def test_bocha_search_does_not_promote_crawl_time_to_publication_time():
    response = FakeResponse({"data": {"webPages": {"value": [{
        "name": "标题", "url": "https://example.com", "snippet": "摘要",
        "dateLastCrawled": "2026-10-16",
    }]}}})
    search = create_bocha_web_search(
        api_key="secret",
        http_client=SimpleNamespace(post=lambda *_args, **_kwargs: response),
    )
    result = search(question="问题", query="查询", evidence=[])[0]

    assert "published_at" not in result
    assert result["crawled_at"] == "2026-10-16"
    assert result["content_origin"] == "search_summary"


def test_bocha_search_handles_provider_error_with_http_200():
    response = FakeResponse({"code": 401, "msg": "invalid key"})
    search = create_bocha_web_search(
        api_key="secret",
        http_client=SimpleNamespace(post=lambda *_args, **_kwargs: response),
    )
    with pytest.raises(RequestError) as exc_info:
        search(question="问题", query="查询", evidence=[])

    assert exc_info.value.status_code == 401
    assert exc_info.value.retryable is False


@pytest.mark.parametrize(
    "status_code, expected_retryable",
    [(400, False), (429, True), (503, True)],
)
def test_bocha_search_http_retry_policy(
    status_code,
    expected_retryable,
):
    error_response = SimpleNamespace(status_code=status_code)
    error = requests.exceptions.HTTPError(
        response=error_response,
    )
    response = FakeResponse({}, status_code=status_code, error=error)
    search = create_bocha_web_search(
        api_key="secret",
        http_client=SimpleNamespace(
            post=lambda *_args, **_kwargs: response
        ),
    )

    with pytest.raises(RequestError) as exc_info:
        search(question="问题", query="查询", evidence=[])

    assert exc_info.value.status_code == status_code
    assert exc_info.value.retryable is expected_retryable
