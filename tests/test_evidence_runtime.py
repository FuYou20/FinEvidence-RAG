from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import threading
from time import sleep

import pytest

from errors import RequestError
import src.evidence_runtime as runtime_module
from src.evidence_cache import RedisEvidenceRequestCache
from src.evidence_runtime import create_evidence_runtime


def metadata_record(chunk_id="c1"):
    return {
        "doc_id": "report_001",
        "page_number": 1,
        "chunk_id": chunk_id,
        "content_type": "text",
        "text": f"证据 {chunk_id}",
    }


def patch_runtime_dependencies(monkeypatch, decisions):
    monkeypatch.setattr(
        runtime_module,
        "build_bm25_index",
        lambda metadata: object(),
    )
    monkeypatch.setattr(
        runtime_module,
        "hybrid_retrieve",
        lambda **_kwargs: [
            {
                **metadata_record(),
                "rank": 1,
                "score": 0.9,
            }
        ],
    )
    monkeypatch.setattr(
        runtime_module,
        "create_evidence_decision_generator",
        lambda *_args, **_kwargs: (
            lambda **_decision_kwargs: next(decisions)
        ),
    )


def make_runtime(
    monkeypatch,
    decisions,
    *,
    search_web=None,
    read_web_page=None,
    max_deepening_steps=2,
    request_cache=None,
):
    patch_runtime_dependencies(monkeypatch, decisions)
    return create_evidence_runtime(
        SimpleNamespace(),
        model_name="test-model",
        dense_index=object(),
        metadata=[metadata_record()],
        embed_batch=lambda texts: [[1.0] for _ in texts],
        search_web=search_web,
        read_web_page=read_web_page,
        neighbor_window=0,
        max_deepening_steps=max_deepening_steps,
        request_cache=request_cache,
    )


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.locks = {}
        self.guard = threading.Lock()

    def get(self, key):
        with self.guard:
            stored = self.values.get(key)
        return None if stored is None else stored[1]

    def setex(self, key, ttl, value):
        with self.guard:
            self.values[key] = (ttl, value)

    def delete(self, key):
        with self.guard:
            self.values.pop(key, None)

    def lock(self, key, **_kwargs):
        with self.guard:
            raw_lock = self.locks.setdefault(key, threading.Lock())

        class FakeLock:
            def acquire(self, *, blocking=True):
                return raw_lock.acquire(blocking=blocking)

            def release(self):
                raw_lock.release()

        return FakeLock()


def test_runtime_registers_local_retrieval_evidence(monkeypatch):
    runtime = make_runtime(
        monkeypatch,
        iter(
            [
                {
                    "enough": True,
                    "next_action": "finish",
                    "reason": "证据已经足够",
                }
            ]
        ),
    )

    bundle = runtime.request_evidence(
        request={
            "request_id": "R1",
            "requester": "support-agent",
            "mode": "stance_research",
            "topic": "辩题",
            "direction": "support",
        }
    )

    evidence_id = bundle.evidence[0]["evidence_id"]
    assert evidence_id.startswith("E-local-")
    assert runtime.registry.get(evidence_id)["chunk_id"] == "c1"
    assert len(runtime.registry) == 1


def test_runtime_reuses_cached_investigation_across_instances(monkeypatch):
    cache = RedisEvidenceRequestCache(
        FakeRedis(),
        namespace="test-index-v1",
    )
    first_runtime = make_runtime(
        monkeypatch,
        iter([{
            "enough": True,
            "next_action": "finish",
            "reason": "证据已经足够",
        }]),
        request_cache=cache,
    )
    first = first_runtime.request_evidence(request={
        "request_id": "R1",
        "requester": "support-agent",
        "mode": "stance_research",
        "topic": "辩题",
        "direction": "support",
    })

    second_runtime = make_runtime(
        monkeypatch,
        iter([]),
        request_cache=cache,
    )
    second = second_runtime.request_evidence(request={
        "request_id": "R2",
        "requester": "judge-agent",
        "mode": "stance_research",
        "topic": "辩题",
        "direction": "support",
    })

    assert second.request_id == "R2"
    assert second.evidence == first.evidence
    assert len(second_runtime.registry) == 1
    assert first_runtime.snapshot_metrics()["evidence_cache_misses"] == 1
    assert first_runtime.snapshot_metrics()["evidence_cache_writes"] == 1
    assert second_runtime.snapshot_metrics()["evidence_cache_hits"] == 1
    assert second_runtime.snapshot_metrics()["avoided_investigations"] == 1
    assert second_runtime.snapshot_requests()[0]["cache"]["status"] == "hit"


def test_distributed_lock_prevents_concurrent_cache_stampede():
    from src.evidence_protocol import EvidenceBundle
    from src.evidence_registry import EvidenceRegistry
    from src.evidence_runtime import EvidenceRuntime

    client = FakeRedis()
    cache = RedisEvidenceRequestCache(client, namespace="stampede-v1")
    handler_calls = 0
    calls_guard = threading.Lock()

    def handler(*, request):
        nonlocal handler_calls
        with calls_guard:
            handler_calls += 1
        sleep(0.05)
        return EvidenceBundle.model_validate({
            "request_id": request.request_id,
            "mode": "stance_research",
            "status": "sufficient",
            "evidence": [{
                "evidence_id": "E-local-shared",
                "source_kind": "local",
                "doc_id": "report-1",
                "chunk_id": "report-1:p1:c0",
                "text": "共享证据",
            }],
        })

    def run_request(position):
        runtime = EvidenceRuntime(
            registry=EvidenceRegistry(),
            request_handler=handler,
            request_cache=cache,
            metrics={
                "evidence_cache_enabled": True,
                "evidence_cache_hits": 0,
                "evidence_cache_misses": 0,
                "evidence_cache_writes": 0,
                "evidence_cache_errors": 0,
                "evidence_cache_lock_waits": 0,
                "evidence_cache_lock_acquired": 0,
                "evidence_cache_post_lock_hits": 0,
                "evidence_cache_lock_errors": 0,
                "avoided_investigations": 0,
            },
        )
        bundle = runtime.request_evidence(request={
            "request_id": f"R-{position}",
            "requester": f"agent-{position}",
            "mode": "stance_research",
            "topic": "同一个并发问题",
            "direction": "support",
        })
        return bundle, runtime.snapshot_metrics(), runtime.snapshot_requests()

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(run_request, range(6)))

    assert handler_calls == 1
    assert sum(item[1]["avoided_investigations"] for item in results) == 5
    assert sum(item[1]["evidence_cache_post_lock_hits"] for item in results) == 5
    assert all(item[0].evidence[0]["evidence_id"] == "E-local-shared" for item in results)
    assert sorted(item[0].request_id for item in results) == [f"R-{i}" for i in range(6)]


def test_runtime_enforces_configured_deepening_budget(monkeypatch):
    runtime = make_runtime(
        monkeypatch,
        iter([{
            "enough": False,
            "next_action": "rewrite_query",
            "reason": "缺少最新信息",
            "next_query": "最新信息",
        }]),
        max_deepening_steps=0,
    )
    bundle = runtime.request_evidence(request={
        "request_id": "R-budget", "requester": "support-agent",
        "mode": "stance_research", "topic": "辩题",
        "direction": "support", "max_deepening_steps": 4,
    })

    assert bundle.status == "insufficient"
    assert bundle.investigation_trace[-1]["stop_reason"] == "max_deepening_steps"
    assert len(bundle.investigation_trace) == 1


def test_verification_starts_from_registered_original_evidence(
    monkeypatch,
):
    decisions = iter(
        [
            {
                "enough": True,
                "next_action": "finish",
                "reason": "初始立场材料足够",
            },
            {
                "enough": True,
                "next_action": "finish",
                "reason": "已核对原始证据",
            },
        ]
    )
    runtime = make_runtime(monkeypatch, decisions)
    stance_bundle = runtime.request_evidence(
        request={
            "request_id": "R1",
            "requester": "support-agent",
            "mode": "stance_research",
            "topic": "辩题",
            "direction": "support",
        }
    )
    evidence_id = stance_bundle.evidence[0]["evidence_id"]

    verification_bundle = runtime.request_evidence(
        request={
            "request_id": "R2",
            "requester": "judge-agent",
            "mode": "verify_evidence",
            "evidence_ids": [evidence_id],
            "verification_goal": "核对原文和口径",
        }
    )

    assert verification_bundle.evidence[0]["evidence_id"] == evidence_id
    assert verification_bundle.evidence[0]["text"] == "证据 c1"


def test_runtime_registers_web_cross_validation_results(monkeypatch):
    decisions = iter(
        [
            {
                "enough": False,
                "next_action": "web_search",
                "reason": "需要独立来源",
                "next_query": "独立来源 最新数据",
            },
            {
                "enough": True,
                "next_action": "finish",
                "reason": "已经获得独立来源",
            },
        ]
    )

    search_queries = []

    def search_web(**kwargs):
        search_queries.append(kwargs["query"])
        return [
            {
                "doc_id": "web:example.com",
                "page_number": 1,
                "chunk_id": "web:1",
                "content_type": "text",
                "text": "独立来源数据",
                "url": "https://example.com/data",
            }
        ]

    runtime = make_runtime(
        monkeypatch,
        decisions,
        search_web=search_web,
    )
    bundle = runtime.request_evidence(
        request={
            "request_id": "R-web",
            "requester": "support-agent",
            "mode": "stance_research",
            "topic": "需要交叉验证的辩题",
            "direction": "support",
        }
    )

    assert bundle.status == "sufficient"
    assert len(bundle.evidence) == 2
    assert bundle.evidence[1]["source_kind"] == "web"
    assert bundle.evidence[1]["evidence_id"].startswith("E-web-")
    assert search_queries == ["独立来源 最新数据"]


def test_runtime_promotes_web_fulltext_and_preserves_search_summary(
    monkeypatch,
):
    decisions = iter([
        {
            "enough": False,
            "next_action": "web_search",
            "reason": "需要网页正文",
            "next_query": "公司 最新市场数据",
        },
        {"enough": True, "next_action": "finish", "reason": "已取得正文"},
    ])

    runtime = make_runtime(
        monkeypatch,
        decisions,
        search_web=lambda **_kwargs: [{
            "doc_id": "web:example.com",
            "page_number": 1,
            "chunk_id": "web:1",
            "content_type": "text",
            "text": "搜索结果摘要",
            "title": "搜索标题",
            "url": "https://example.com/redirect",
            "content_origin": "search_summary",
        }],
        read_web_page=lambda *, url: {
            "url": "https://example.com/article",
            "title": "网页正式标题",
            "published_at": "2026-10-01",
            "text": "这是从网页中抽取的完整正文，包含可核对的数据和口径。",
            "truncated": False,
        },
    )
    bundle = runtime.request_evidence(request={
        "request_id": "R-fulltext",
        "requester": "support-agent",
        "mode": "stance_research",
        "topic": "需要全文交叉验证的辩题",
        "direction": "support",
    })

    web = bundle.evidence[1]
    metrics = runtime.snapshot_metrics()
    assert web["text"].startswith("这是从网页中抽取的完整正文")
    assert web["content_origin"] == "web_fulltext"
    assert web["search_summary"] == "搜索结果摘要"
    assert web["url"] == "https://example.com/article"
    assert web["search_result_url"] == "https://example.com/redirect"
    assert web["doc_id"] == "web:example.com"
    assert web["published_at"] == "2026-10-01"
    assert metrics["web_page_read_calls"] == 1
    assert metrics["web_page_read_successes"] == 1
    assert metrics["search_summary_fallbacks"] == 0


def test_runtime_keeps_summary_when_page_read_fails(monkeypatch):
    decisions = iter([
        {
            "enough": False,
            "next_action": "web_search",
            "reason": "需要外部信息",
            "next_query": "外部信息",
        },
        {"enough": True, "next_action": "finish", "reason": "摘要可用"},
    ])

    def unavailable_page_reader(*, url):
        raise RequestError("网页拒绝访问", retryable=False, status_code=403)

    runtime = make_runtime(
        monkeypatch,
        decisions,
        search_web=lambda **_kwargs: [{
            "doc_id": "web:example.com",
            "page_number": 1,
            "chunk_id": "web:1",
            "content_type": "text",
            "text": "仍可使用的搜索摘要",
            "url": "https://example.com/paywall",
            "content_origin": "search_summary",
        }],
        read_web_page=unavailable_page_reader,
    )
    bundle = runtime.request_evidence(request={
        "request_id": "R-summary-fallback",
        "requester": "support-agent",
        "mode": "stance_research",
        "topic": "需要外部信息",
        "direction": "support",
    })

    web = bundle.evidence[1]
    metrics = runtime.snapshot_metrics()
    assert web["text"] == "仍可使用的搜索摘要"
    assert web["content_origin"] == "search_summary"
    assert web["extraction_status"] == "failed"
    assert web["extraction_retryable"] is False
    assert metrics["web_page_read_failures"] == 1
    assert metrics["search_summary_fallbacks"] == 1


def test_runtime_does_not_hide_page_reader_programming_errors(monkeypatch):
    decisions = iter([{
        "enough": False,
        "next_action": "web_search",
        "reason": "需要外部信息",
        "next_query": "外部信息",
    }])
    runtime = make_runtime(
        monkeypatch,
        decisions,
        search_web=lambda **_kwargs: [{
            "text": "摘要",
            "url": "https://example.com",
        }],
        read_web_page=lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("reader bug")
        ),
    )

    with pytest.raises(RuntimeError, match="reader bug"):
        runtime.request_evidence(request={
            "request_id": "R-reader-bug",
            "requester": "support-agent",
            "mode": "stance_research",
            "topic": "辩题",
            "direction": "support",
        })


def test_runtime_falls_back_to_bm25_and_opens_dense_circuit(monkeypatch):
    class DenseTimeout(Exception):
        pass

    hybrid_calls = []
    sparse_calls = []
    decisions = iter([
        {"enough": True, "next_action": "finish", "reason": "足够"},
        {"enough": True, "next_action": "finish", "reason": "足够"},
    ])
    patch_runtime_dependencies(monkeypatch, decisions)
    monkeypatch.setattr(
        runtime_module,
        "TRANSIENT_DENSE_ERRORS",
        (DenseTimeout,),
    )
    monkeypatch.setattr(
        runtime_module,
        "hybrid_retrieve",
        lambda **kwargs: (
            hybrid_calls.append(kwargs),
            (_ for _ in ()).throw(DenseTimeout()),
        )[1],
    )
    monkeypatch.setattr(
        runtime_module,
        "sparse_retrieve",
        lambda **kwargs: (
            sparse_calls.append(kwargs)
            or [{**metadata_record(), "rank": 1, "score": 1.0}]
        ),
    )
    runtime = create_evidence_runtime(
        SimpleNamespace(),
        model_name="test-model",
        dense_index=object(),
        metadata=[metadata_record()],
        embed_batch=lambda texts: [[1.0] for _ in texts],
        neighbor_window=0,
    )

    for request_id in ("R-fallback-1", "R-fallback-2"):
        runtime.request_evidence(request={
            "request_id": request_id,
            "requester": "support-agent",
            "mode": "stance_research",
            "topic": "辩题",
            "direction": "support",
        })

    metrics = runtime.snapshot_metrics()
    assert len(hybrid_calls) == 1
    assert len(sparse_calls) == 2
    assert metrics["degraded_retrievals"] == 2
    assert metrics["dense_circuit_open"] is True
    assert metrics["last_dense_error_type"] == "DenseTimeout"
