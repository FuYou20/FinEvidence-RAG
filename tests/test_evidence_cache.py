import json
import threading

import pytest

from src.evidence_cache import (
    EvidenceCachePayloadError,
    RedisEvidenceRequestCache,
    build_cache_namespace,
    build_evidence_request_cache_key,
)


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.deleted = []
        self.locks = {}

    def get(self, key):
        stored = self.values.get(key)
        return None if stored is None else stored[1]

    def setex(self, key, ttl, value):
        self.values[key] = (ttl, value)
        return True

    def delete(self, key):
        self.deleted.append(key)
        self.values.pop(key, None)
        return 1

    def lock(self, key, **_kwargs):
        raw_lock = self.locks.setdefault(key, threading.Lock())

        class FakeLock:
            def acquire(self, *, blocking=True):
                return raw_lock.acquire(blocking=blocking)

            def release(self):
                raw_lock.release()

        return FakeLock()


def stance_request(request_id="R1", requester="support-agent"):
    return {
        "request_id": request_id,
        "requester": requester,
        "mode": "stance_research",
        "topic": "竞争优势是否持续",
        "direction": "support",
        "research_focus": ["渠道", "产品"],
        "context_evidence_ids": ["E2", "E1"],
        "max_deepening_steps": 2,
    }


def sufficient_bundle(request_id="R1"):
    return {
        "request_id": request_id,
        "mode": "stance_research",
        "status": "sufficient",
        "evidence": [{
            "evidence_id": "E-local-1",
            "source_kind": "local",
            "doc_id": "report-1",
            "chunk_id": "report-1:p1:c0",
            "text": "证据正文",
        }],
    }


def test_cache_key_ignores_request_identity_and_list_order():
    first = stance_request()
    second = stance_request("R2", "judge-agent")
    second["research_focus"] = ["产品", "渠道"]
    second["context_evidence_ids"] = ["E1", "E2"]

    first_key = build_evidence_request_cache_key(first, namespace="index-v1")
    second_key = build_evidence_request_cache_key(second, namespace="index-v1")

    assert first_key == second_key


def test_cache_namespace_changes_when_index_config_changes():
    first = build_cache_namespace({"corpus": "v1", "top_k": 8})
    second = build_cache_namespace({"corpus": "v2", "top_k": 8})

    assert first != second


def test_cache_round_trip_rebinds_request_id_and_uses_sufficient_ttl():
    client = FakeRedis()
    cache = RedisEvidenceRequestCache(
        client,
        namespace="index-v1",
        ttl_seconds=600,
        insufficient_ttl_seconds=30,
    )
    request = stance_request()

    key = cache.set(request, sufficient_bundle())
    restored = cache.get(stance_request("R-new", "oppose-agent"))

    assert client.values[key][0] == 600
    assert restored.request_id == "R-new"
    assert restored.evidence[0]["evidence_id"] == "E-local-1"


def test_insufficient_result_uses_short_ttl():
    client = FakeRedis()
    cache = RedisEvidenceRequestCache(
        client,
        namespace="index-v1",
        ttl_seconds=600,
        insufficient_ttl_seconds=30,
    )
    bundle = {
        "request_id": "R1",
        "mode": "stance_research",
        "status": "insufficient",
        "missing_information": ["缺少最新数据"],
    }

    key = cache.set(stance_request(), bundle)

    assert client.values[key][0] == 30


def test_corrupt_cache_entry_is_deleted_and_reported():
    client = FakeRedis()
    cache = RedisEvidenceRequestCache(client, namespace="index-v1")
    request = stance_request()
    key = cache.make_key(request)
    client.values[key] = (60, json.dumps({"schema_version": 999}))

    with pytest.raises(EvidenceCachePayloadError, match="内容损坏"):
        cache.get(request)

    assert key in client.deleted
    assert key not in client.values


def test_distributed_lock_uses_request_specific_key():
    client = FakeRedis()
    cache = RedisEvidenceRequestCache(client, namespace="index-v1")
    request = stance_request()

    with cache.lock(request):
        lock_key = cache.make_lock_key(request)
        assert lock_key.endswith(":lock")
        assert client.locks[lock_key].locked()

    assert not client.locks[lock_key].locked()
