"""Redis-backed cache for completed evidence requests.

The cache stores validated ``EvidenceBundle`` objects.  It is deliberately
separate from ``EvidenceRegistry``: the registry deduplicates evidence inside
one Python process, while this cache reuses completed investigations across
agents and process restarts.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager

from .evidence_protocol import EvidenceBundle, EvidenceRequest


CACHE_SCHEMA_VERSION = 1


class EvidenceCacheError(RuntimeError):
    """Base class for cache failures that may safely degrade to a miss."""


class EvidenceCacheUnavailable(EvidenceCacheError):
    """Redis cannot currently serve the request."""


class EvidenceCachePayloadError(EvidenceCacheError):
    """A cached value cannot be decoded or validated."""


def _positive_int(value, *, field_name):
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ValueError(f"{field_name} 必须是正整数")
    return value


def _nonempty_string(value, *, field_name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} 必须是非空字符串")
    return value.strip()


def _canonical_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_cache_namespace(config):
    """Create an index/config fingerprint used for cache invalidation."""
    if not isinstance(config, dict) or not config:
        raise ValueError("config 必须是非空 dict")
    digest = hashlib.sha256(
        _canonical_json(config).encode("utf-8")
    ).hexdigest()
    return f"cfg-{digest[:20]}"


def build_evidence_request_cache_key(
    request,
    *,
    namespace,
    key_prefix="evidence-request",
):
    """Build a stable key while ignoring caller-specific request metadata."""
    validated = EvidenceRequest.model_validate(request)
    namespace = _nonempty_string(namespace, field_name="namespace")
    key_prefix = _nonempty_string(key_prefix, field_name="key_prefix")
    payload = validated.model_dump(mode="json")
    # Different agents and request IDs may ask the same semantic question.
    payload.pop("request_id", None)
    payload.pop("requester", None)
    for field in (
        "research_focus",
        "context_evidence_ids",
        "evidence_ids",
    ):
        payload[field] = sorted(payload.get(field, []))
    digest = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return f"{key_prefix}:{namespace}:{digest}"


class RedisEvidenceRequestCache:
    """Store evidence bundles in a Redis-compatible client.

    The injected client only needs ``get``, ``setex``, ``delete`` and
    ``lock``.  The distributed lock prevents concurrent cache misses from
    triggering the same expensive investigation repeatedly.
    """

    def __init__(
        self,
        client,
        *,
        namespace,
        ttl_seconds=21_600,
        insufficient_ttl_seconds=900,
        lock_timeout_seconds=180,
        lock_blocking_timeout_seconds=185,
        key_prefix="evidence-request",
    ):
        for method_name in ("get", "setex", "delete", "lock"):
            if not callable(getattr(client, method_name, None)):
                raise TypeError(
                    f"Redis client 缺少可调用方法: {method_name}"
                )
        self._client = client
        self.namespace = _nonempty_string(
            namespace,
            field_name="namespace",
        )
        self.ttl_seconds = _positive_int(
            ttl_seconds,
            field_name="ttl_seconds",
        )
        self.insufficient_ttl_seconds = _positive_int(
            insufficient_ttl_seconds,
            field_name="insufficient_ttl_seconds",
        )
        self.lock_timeout_seconds = _positive_int(
            lock_timeout_seconds,
            field_name="lock_timeout_seconds",
        )
        self.lock_blocking_timeout_seconds = _positive_int(
            lock_blocking_timeout_seconds,
            field_name="lock_blocking_timeout_seconds",
        )
        self.key_prefix = _nonempty_string(
            key_prefix,
            field_name="key_prefix",
        )

    @classmethod
    def from_url(cls, url, **kwargs):
        """Create a cache from ``REDIS_URL`` without importing Redis eagerly."""
        url = _nonempty_string(url, field_name="url")
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError(
                "缺少 redis Python 包，请先安装 requirements.txt"
            ) from exc
        client_options = {
            "decode_responses": True,
            "socket_connect_timeout": 1.5,
            "socket_timeout": 2.0,
            "health_check_interval": 30,
        }
        client_options.update(kwargs.pop("client_options", {}))
        client = redis.Redis.from_url(url, **client_options)
        return cls(client, **kwargs)

    def make_key(self, request):
        return build_evidence_request_cache_key(
            request,
            namespace=self.namespace,
            key_prefix=self.key_prefix,
        )

    def make_lock_key(self, request):
        return f"{self.make_key(request)}:lock"

    @contextmanager
    def lock(self, request):
        """Serialize builders for one semantic request across processes.

        The lock has a TTL so a crashed worker cannot leave a permanent lock.
        Release failures are intentionally ignored: the protected result has
        already been produced and Redis will remove the lock at its TTL.
        """
        try:
            distributed_lock = self._client.lock(
                self.make_lock_key(request),
                timeout=self.lock_timeout_seconds,
                blocking_timeout=self.lock_blocking_timeout_seconds,
                thread_local=False,
            )
            acquired = distributed_lock.acquire(blocking=True)
        except Exception as exc:
            self._raise_if_redis_error(exc)
            raise
        if not acquired:
            raise EvidenceCacheUnavailable(
                "等待 Redis 证据缓存锁超时"
            )
        try:
            yield
        finally:
            try:
                distributed_lock.release()
            except Exception:
                # The lock TTL is the final safety net.  Never discard a
                # successful investigation because lock cleanup failed.
                pass

    def ping(self):
        ping = getattr(self._client, "ping", None)
        if not callable(ping):
            raise TypeError("Redis client 缺少可调用方法: ping")
        try:
            return bool(ping())
        except Exception as exc:
            self._raise_if_redis_error(exc)
            raise

    def close(self):
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def get(self, request):
        key = self.make_key(request)
        try:
            raw = self._client.get(key)
        except Exception as exc:
            self._raise_if_redis_error(exc)
            raise
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            payload = json.loads(raw)
            if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
                raise ValueError("缓存 schema_version 不匹配")
            bundle = EvidenceBundle.model_validate(payload["bundle"])
        except (KeyError, TypeError, ValueError) as exc:
            try:
                self._client.delete(key)
            except Exception as delete_exc:
                self._raise_if_redis_error(delete_exc)
                raise
            raise EvidenceCachePayloadError(
                f"Redis 证据缓存内容损坏: {key}"
            ) from exc
        request_id = EvidenceRequest.model_validate(request).request_id
        return bundle.model_copy(update={"request_id": request_id})

    def set(self, request, bundle):
        validated_bundle = EvidenceBundle.model_validate(bundle)
        key = self.make_key(request)
        ttl = (
            self.ttl_seconds
            if validated_bundle.status == "sufficient"
            else self.insufficient_ttl_seconds
        )
        payload = _canonical_json({
            "schema_version": CACHE_SCHEMA_VERSION,
            "bundle": validated_bundle.model_dump(mode="json"),
        })
        try:
            self._client.setex(key, ttl, payload)
        except Exception as exc:
            self._raise_if_redis_error(exc)
            raise
        return key

    @staticmethod
    def _raise_if_redis_error(exc):
        try:
            import redis
        except ImportError:
            return
        if isinstance(exc, redis.exceptions.RedisError):
            raise EvidenceCacheUnavailable(
                f"Redis 证据缓存暂时不可用: {type(exc).__name__}"
            ) from exc
