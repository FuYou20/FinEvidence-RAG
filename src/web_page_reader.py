"""Fetch and extract readable text from public web pages.

Search-provider snippets remain the fallback.  This module deliberately owns
network safety, response-size limits and HTML-to-text conversion so the
evidence runtime does not treat an arbitrary URL as trusted input.
"""

from __future__ import annotations

from html.parser import HTMLParser
import ipaddress
import re
import socket
from urllib.parse import urldefrag, urljoin, urlparse

import requests

from errors import RequestError


_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_SKIP_TAGS = {
    "script",
    "style",
    "noscript",
    "svg",
    "canvas",
    "nav",
    "footer",
    "header",
    "form",
}
_BLOCK_TAGS = {
    "article",
    "aside",
    "blockquote",
    "br",
    "div",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "main",
    "p",
    "section",
    "table",
    "tr",
}
_PUBLISHED_KEYS = {
    "article:published_time",
    "date",
    "datepublished",
    "publishdate",
    "pubdate",
}


def _normalize_required_string(value, *, field_name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} 必须是非空字符串")
    return value.strip()


def _is_public_ip(address):
    try:
        return ipaddress.ip_address(address).is_global
    except ValueError:
        return False


def validate_public_web_url(url, *, resolve_host=socket.getaddrinfo):
    """Return a normalized URL after rejecting local/private destinations."""
    normalized = _normalize_required_string(url, field_name="url")
    normalized, _fragment = urldefrag(normalized)
    parsed = urlparse(normalized)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise RequestError("网页 URL 只允许 http 或 https")
    if not parsed.hostname:
        raise RequestError("网页 URL 缺少主机名")
    if parsed.username is not None or parsed.password is not None:
        raise RequestError("网页 URL 不允许包含用户凭据")
    try:
        explicit_port = parsed.port
    except ValueError as exc:
        raise RequestError("网页 URL 端口不合法") from exc
    if explicit_port is not None and explicit_port not in {80, 443}:
        raise RequestError("网页 URL 只允许标准 HTTP/HTTPS 端口")

    host = parsed.hostname
    try:
        literal_ip = ipaddress.ip_address(host)
    except ValueError:
        literal_ip = None
    if literal_ip is not None:
        addresses = [str(literal_ip)]
    else:
        try:
            addresses = {
                item[4][0]
                for item in resolve_host(
                    host,
                    explicit_port or (443 if parsed.scheme == "https" else 80),
                    type=socket.SOCK_STREAM,
                )
            }
        except (OSError, socket.gaierror) as exc:
            raise RequestError(
                "网页域名解析失败",
                retryable=True,
            ) from exc
    if not addresses or any(not _is_public_ip(item) for item in addresses):
        raise RequestError("网页 URL 指向了非公网地址")
    return normalized


class _ReadableHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._skip_stack = []
        self._content_depth = 0
        self._in_title = False
        self._all_parts = []
        self._main_parts = []
        self._title_parts = []
        self.published_at = None

    def _append(self, value):
        if self._skip_stack:
            return
        self._all_parts.append(value)
        if self._content_depth:
            self._main_parts.append(value)

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        attributes = {
            str(key).lower(): value
            for key, value in attrs
            if key is not None
        }
        if self._skip_stack:
            if tag in _SKIP_TAGS:
                self._skip_stack.append(tag)
            return
        if tag in _SKIP_TAGS:
            self._skip_stack.append(tag)
            return
        if tag in {"main", "article"}:
            self._content_depth += 1
        if tag == "title":
            self._in_title = True
        if tag == "meta" and self.published_at is None:
            key = (
                attributes.get("property")
                or attributes.get("name")
                or attributes.get("itemprop")
                or ""
            ).lower()
            content = attributes.get("content")
            if key in _PUBLISHED_KEYS and content and content.strip():
                self.published_at = content.strip()
        if tag == "time" and self.published_at is None:
            timestamp = attributes.get("datetime")
            if timestamp and timestamp.strip():
                self.published_at = timestamp.strip()
        if tag in _BLOCK_TAGS:
            self._append("\n")
        elif tag in {"td", "th"}:
            self._append("\t")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if self._skip_stack:
            if tag == self._skip_stack[-1]:
                self._skip_stack.pop()
            return
        if tag == "title":
            self._in_title = False
        if tag in _BLOCK_TAGS:
            self._append("\n")
        if tag in {"main", "article"} and self._content_depth:
            self._content_depth -= 1

    def handle_data(self, data):
        if self._skip_stack:
            return
        if self._in_title:
            self._title_parts.append(data)
        self._append(data)

    @staticmethod
    def _clean(parts):
        text = "".join(parts).replace("\xa0", " ")
        lines = []
        previous = None
        for raw_line in text.splitlines():
            line = re.sub(r"[ \t\r\f\v]+", " ", raw_line).strip()
            if line and line != previous:
                lines.append(line)
                previous = line
        return "\n".join(lines)

    def result(self):
        main_text = self._clean(self._main_parts)
        all_text = self._clean(self._all_parts)
        return {
            "title": self._clean(self._title_parts) or None,
            "text": main_text if len(main_text) >= 80 else all_text,
            "published_at": self.published_at,
        }


def _read_limited_content(response, max_content_bytes):
    headers = getattr(response, "headers", {}) or {}
    content_length = headers.get("Content-Length") or headers.get(
        "content-length"
    )
    if content_length is not None:
        try:
            if int(content_length) > max_content_bytes:
                raise RequestError("网页响应超过允许的大小")
        except ValueError:
            pass

    iter_content = getattr(response, "iter_content", None)
    if callable(iter_content):
        chunks = []
        total = 0
        for chunk in iter_content(chunk_size=16_384):
            if not chunk:
                continue
            total += len(chunk)
            if total > max_content_bytes:
                raise RequestError("网页响应超过允许的大小")
            chunks.append(chunk)
        return b"".join(chunks)

    content = getattr(response, "content", b"")
    if isinstance(content, str):
        content = content.encode("utf-8")
    if len(content) > max_content_bytes:
        raise RequestError("网页响应超过允许的大小")
    return content


def _decode_response(response, content):
    encoding = getattr(response, "encoding", None)
    if not encoding or encoding.lower() in {"iso-8859-1", "latin-1"}:
        encoding = getattr(response, "apparent_encoding", None) or "utf-8"
    try:
        return content.decode(encoding, errors="replace")
    except LookupError:
        return content.decode("utf-8", errors="replace")


def create_web_page_reader(
    *,
    timeout=10.0,
    max_content_bytes=2_000_000,
    max_text_chars=30_000,
    min_text_chars=200,
    max_redirects=3,
    http_client=requests,
    resolve_host=socket.getaddrinfo,
):
    """Create a callable that safely turns one public URL into evidence text."""
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("timeout 必须是正数")
    for name, value in (
        ("max_content_bytes", max_content_bytes),
        ("max_text_chars", max_text_chars),
        ("min_text_chars", min_text_chars),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} 必须是正整数")
    if (
        not isinstance(max_redirects, int)
        or isinstance(max_redirects, bool)
        or max_redirects < 0
    ):
        raise ValueError("max_redirects 必须是非负整数")
    get = getattr(http_client, "get", None)
    if not callable(get):
        raise TypeError("http_client 必须提供可调用的 get()")

    def read_web_page(*, url):
        current_url = validate_public_web_url(
            url,
            resolve_host=resolve_host,
        )
        original_url = current_url
        followed_redirects = 0
        response = None
        try:
            for redirect_count in range(max_redirects + 1):
                response = get(
                    current_url,
                    headers={
                        "User-Agent": "FinEvidenceBot/1.0 (+research evidence reader)",
                        "Accept": "text/html,text/plain;q=0.9",
                    },
                    timeout=float(timeout),
                    allow_redirects=False,
                    stream=True,
                )
                status_code = getattr(response, "status_code", None)
                if status_code in _REDIRECT_STATUSES:
                    if redirect_count >= max_redirects:
                        raise RequestError("网页重定向次数超过限制")
                    location = (getattr(response, "headers", {}) or {}).get(
                        "Location"
                    )
                    if not location:
                        raise RequestError("网页重定向缺少 Location")
                    current_url = validate_public_web_url(
                        urljoin(current_url, location),
                        resolve_host=resolve_host,
                    )
                    followed_redirects += 1
                    close = getattr(response, "close", None)
                    if callable(close):
                        close()
                    response = None
                    continue
                response.raise_for_status()
                break

            original_path = urlparse(original_url).path.rstrip("/")
            final_path = urlparse(current_url).path.rstrip("/")
            if original_path and not final_path:
                raise RequestError(
                    "候选文章被重定向到站点首页，不能视为正文证据"
                )

            headers = getattr(response, "headers", {}) or {}
            content_type = (
                headers.get("Content-Type")
                or headers.get("content-type")
                or ""
            ).lower()
            if content_type and not any(
                supported in content_type
                for supported in ("text/html", "application/xhtml+xml", "text/plain")
            ):
                raise RequestError(f"不支持的网页内容类型: {content_type}")
            content = _read_limited_content(response, max_content_bytes)
            decoded = _decode_response(response, content)
            if "text/plain" in content_type:
                extracted = {
                    "title": None,
                    "text": decoded.strip(),
                    "published_at": None,
                }
            else:
                parser = _ReadableHTMLParser()
                parser.feed(decoded)
                extracted = parser.result()
            text = extracted["text"].strip()
            if len(text) < min_text_chars:
                raise RequestError("网页正文过短，无法作为全文证据")
            return {
                "url": current_url,
                "title": extracted["title"],
                "published_at": extracted["published_at"],
                "text": text[:max_text_chars],
                "truncated": len(text) > max_text_chars,
                "redirect_count": followed_redirects,
                "content_origin": "web_fulltext",
                "extraction_status": "success",
            }
        except requests.exceptions.Timeout as exc:
            raise RequestError(
                "读取网页正文超时",
                retryable=True,
                status_code=408,
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise RequestError(
                "读取网页正文连接失败",
                retryable=True,
                status_code=502,
            ) from exc
        except requests.exceptions.HTTPError as exc:
            status_code = getattr(exc.response, "status_code", None)
            retryable = status_code == 429 or (
                status_code is not None and status_code >= 500
            )
            raise RequestError(
                f"读取网页正文 HTTP 错误: {status_code}",
                retryable=retryable,
                status_code=status_code,
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RequestError(
                "读取网页正文请求失败",
                retryable=True,
            ) from exc
        finally:
            if response is not None:
                close = getattr(response, "close", None)
                if callable(close):
                    close()

    return read_web_page
