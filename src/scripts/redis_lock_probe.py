"""Exercise the evidence cache lock against a real Redis instance."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from threading import Lock
from time import sleep
from uuid import uuid4

from dotenv import load_dotenv

from src.evidence_cache import RedisEvidenceRequestCache
from src.evidence_protocol import EvidenceBundle
from src.evidence_registry import EvidenceRegistry
from src.evidence_runtime import EvidenceRuntime


def _empty_metrics():
    return {
        "evidence_requests": 0,
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
    }


def main():
    load_dotenv()
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        raise RuntimeError("缺少 REDIS_URL")
    cache = RedisEvidenceRequestCache.from_url(
        redis_url,
        namespace=f"lock-probe-{uuid4().hex}",
        ttl_seconds=60,
        insufficient_ttl_seconds=30,
        lock_timeout_seconds=10,
        lock_blocking_timeout_seconds=15,
    )
    handler_calls = 0
    calls_guard = Lock()

    def handler(*, request):
        nonlocal handler_calls
        with calls_guard:
            handler_calls += 1
        sleep(1.0)
        return EvidenceBundle.model_validate({
            "request_id": request.request_id,
            "mode": request.mode,
            "status": "sufficient",
            "evidence": [{
                "evidence_id": "E-local-lock-probe",
                "source_kind": "local",
                "doc_id": "lock-probe",
                "chunk_id": "lock-probe:p1:c0",
                "text": "由唯一一次调查生成的证据",
            }],
        })

    def request_as_agent(position):
        runtime = EvidenceRuntime(
            registry=EvidenceRegistry(),
            request_handler=handler,
            request_cache=cache,
            metrics=_empty_metrics(),
        )
        bundle = runtime.request_evidence(request={
            "request_id": f"probe-{position}",
            "requester": f"agent-{position}",
            "mode": "stance_research",
            "topic": "相同的并发证据请求",
            "direction": "support",
        })
        return {
            "request_id": bundle.request_id,
            "metrics": runtime.snapshot_metrics(),
        }

    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(request_as_agent, range(8)))
    finally:
        cache.close()
    summary = {
        "concurrent_agents": len(results),
        "actual_handler_calls": handler_calls,
        "avoided_investigations": sum(
            item["metrics"]["avoided_investigations"]
            for item in results
        ),
        "post_lock_hits": sum(
            item["metrics"]["evidence_cache_post_lock_hits"]
            for item in results
        ),
        "initial_hits": sum(
            item["metrics"]["evidence_cache_hits"]
            - item["metrics"]["evidence_cache_post_lock_hits"]
            for item in results
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if handler_calls != 1 or summary["avoided_investigations"] != 7:
        raise RuntimeError("分布式锁未能阻止缓存击穿")


if __name__ == "__main__":
    main()
