import hashlib
from urllib.parse import urlparse

import requests

from errors import RequestError


DEFAULT_BOCHA_ENDPOINT = "https://api.bocha.cn/v1/web-search"


def _normalize_required_string(value, *, field_name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} 必须是非空字符串")
    return value.strip()


def _normalize_web_page(page):
    if not isinstance(page, dict):
        raise RequestError("博查返回的网页记录不是 JSON 对象")

    url = _normalize_required_string(page.get("url"), field_name="url")
    title = page.get("name") or page.get("title") or url
    title = _normalize_required_string(title, field_name="title")
    text = page.get("summary") or page.get("snippet") or title
    text = _normalize_required_string(text, field_name="snippet")
    host = urlparse(url).netloc or "web"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]

    record = {
        "doc_id": f"web:{host}",
        "page_number": 1,
        "chunk_id": f"web:{digest}",
        "content_type": "text",
        "text": text,
        "title": title,
        "url": url,
        "source": title,
        "content_origin": "search_summary",
    }
    published_at = page.get("datePublished") or page.get("publishedAt")
    if isinstance(published_at, str) and published_at.strip():
        record["published_at"] = published_at.strip()
    crawled_at = page.get("dateLastCrawled")
    if isinstance(crawled_at, str) and crawled_at.strip():
        record["crawled_at"] = crawled_at.strip()
    return record


def create_bocha_web_search(
    *,
    api_key,
    endpoint=DEFAULT_BOCHA_ENDPOINT,
    count=5,
    timeout=30.0,
    http_client=requests,
):
    """创建符合证据调查工作流签名的博查搜索函数。"""
    normalized_api_key = _normalize_required_string(
        api_key,
        field_name="api_key",
    )
    normalized_endpoint = _normalize_required_string(
        endpoint,
        field_name="endpoint",
    )
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 50:
        raise ValueError("count 必须在 1 到 50 之间")
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("timeout 必须是正数")
    post = getattr(http_client, "post", None)
    if not callable(post):
        raise TypeError("http_client 必须提供可调用的 post()")

    def search_web(*, question, query, evidence):
        del question, evidence
        normalized_query = _normalize_required_string(
            query,
            field_name="query",
        )
        try:
            response = post(
                normalized_endpoint,
                headers={
                    "Authorization": f"Bearer {normalized_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "query": normalized_query,
                    "summary": True,
                    "count": count,
                },
                timeout=float(timeout),
            )
            response.raise_for_status()
            try:
                payload = response.json()
                if not isinstance(payload, dict):
                    raise TypeError("response JSON must be an object")
                provider_code = payload.get("code", 200)
                if provider_code not in (200, "200"):
                    numeric_code = (
                        int(provider_code)
                        if str(provider_code).isdigit()
                        else None
                    )
                    raise RequestError(
                        f"博查搜索业务错误: {provider_code}",
                        retryable=numeric_code == 429 or (
                            numeric_code is not None and numeric_code >= 500
                        ),
                        status_code=numeric_code,
                    )
                pages = payload["data"]["webPages"]["value"]
            except (KeyError, TypeError, ValueError) as exc:
                raise RequestError(
                    "博查返回结构不符合预期",
                    retryable=True,
                    status_code=getattr(response, "status_code", None),
                ) from exc
            if not isinstance(pages, list):
                raise RequestError(
                    "博查 webPages.value 必须是列表",
                    retryable=True,
                    status_code=getattr(response, "status_code", None),
                )
            try:
                return [_normalize_web_page(page) for page in pages]
            except ValueError as exc:
                raise RequestError(
                    "博查网页记录缺少必要字段",
                    retryable=True,
                    status_code=getattr(response, "status_code", None),
                ) from exc
        except requests.exceptions.Timeout as exc:
            raise RequestError(
                "博查搜索请求超时",
                retryable=True,
                status_code=408,
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise RequestError(
                "博查搜索连接失败",
                retryable=True,
                status_code=502,
            ) from exc
        except requests.exceptions.HTTPError as exc:
            status_code = getattr(exc.response, "status_code", None)
            retryable = status_code == 429 or (
                status_code is not None and status_code >= 500
            )
            raise RequestError(
                f"博查搜索 HTTP 错误: {status_code}",
                retryable=retryable,
                status_code=status_code,
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RequestError(
                "博查搜索请求失败",
                retryable=True,
            ) from exc

    return search_web
