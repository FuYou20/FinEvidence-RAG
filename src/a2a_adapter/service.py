"""Assemble the production EvidenceRuntime used by the A2A server."""

import logging
import os
from functools import partial
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "retrieval_index"
CORPUS_PATH = PROJECT_ROOT / "data" / "eval" / "corpus_v1.jsonl"


def _require_env(name):
    value = os.getenv(name)
    if not value or not value.strip():
        raise RuntimeError(f"未检测到 {name}")
    return value.strip()


def _positive_int_env(name, default):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是正整数") from exc
    if value <= 0:
        raise ValueError(f"{name} 必须是正整数")
    return value


def _bool_env(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} 必须是 true/false")


class EvidenceService:
    def __init__(self, runtime, *, request_cache=None):
        self.runtime = runtime
        self.request_cache = request_cache

    def request_evidence(self, *, request):
        return self.runtime.request_evidence(request=request)

    def close(self):
        if self.request_cache is not None:
            self.request_cache.close()


def create_evidence_service_from_env(*, enable_web=None):
    """Load the persisted index and build one long-lived evidence service."""
    load_dotenv(PROJECT_ROOT / ".env")
    # Import after dotenv because embedder initializes its client at import.
    from src.corpus_indexer import load_corpus_manifest
    from src.embedder import qwen_embed_batch
    from src.evidence_cache import (
        EvidenceCacheError,
        RedisEvidenceRequestCache,
        build_cache_namespace,
    )
    from src.evidence_runtime import create_evidence_runtime
    from src.index_cache import load_index_bundle
    from src.web_page_reader import create_web_page_reader
    from src.web_search import create_bocha_web_search

    model_name = _require_env("DEEPSEEK_MODEL")
    client = OpenAI(
        api_key=_require_env("DEEPSEEK_API_KEY"),
        base_url=_require_env("DEEPSEEK_BASE_URL"),
        timeout=60.0,
        max_retries=1,
    )
    dense_index, metadata, build_config = load_index_bundle(CACHE_DIR)
    corpus_records = load_corpus_manifest(CORPUS_PATH)
    embed_batch = partial(
        qwen_embed_batch,
        model_name=build_config["embedding_model"],
    )
    web_enabled = (
        _bool_env("A2A_ENABLE_WEB", False)
        if enable_web is None
        else bool(enable_web)
    )
    search_web = None
    read_web_page = None
    if web_enabled:
        search_web = create_bocha_web_search(
            api_key=_require_env("BOCHA_API_KEY")
        )
        read_web_page = create_web_page_reader()
    retrieval_config = {
        "top_k": 8,
        "dense_k": 20,
        "sparse_k": 20,
        "neighbor_window": 1,
        "neighbor_seed_top_n": 3,
        "neighbor_decay": 0.8,
    }
    max_deepening_steps = _positive_int_env(
        "A2A_MAX_DEEPENING_STEPS",
        2,
    )
    request_cache = None
    redis_url = os.getenv("REDIS_URL")
    if redis_url and redis_url.strip():
        namespace = build_cache_namespace({
            "runtime_policy_version": "evidence-runtime-v1",
            "service": "finevidence-a2a-v1",
            "index_build_config": build_config,
            "retrieval_config": retrieval_config,
            "model_name": model_name,
            "max_deepening_steps": max_deepening_steps,
            "web_enabled": web_enabled,
        })
        try:
            request_cache = RedisEvidenceRequestCache.from_url(
                redis_url.strip(),
                namespace=namespace,
                ttl_seconds=_positive_int_env(
                    "EVIDENCE_CACHE_TTL_SECONDS", 21_600
                ),
                insufficient_ttl_seconds=_positive_int_env(
                    "EVIDENCE_CACHE_INSUFFICIENT_TTL_SECONDS", 900
                ),
                lock_timeout_seconds=_positive_int_env(
                    "EVIDENCE_CACHE_LOCK_TIMEOUT_SECONDS", 180
                ),
                lock_blocking_timeout_seconds=_positive_int_env(
                    "EVIDENCE_CACHE_LOCK_BLOCKING_TIMEOUT_SECONDS", 185
                ),
            )
            request_cache.ping()
        except (EvidenceCacheError, RuntimeError) as exc:
            logger.warning("Redis 不可用，A2A 服务降级为实时调查: %s", exc)
            request_cache = None
    runtime = create_evidence_runtime(
        client,
        model_name=model_name,
        dense_index=dense_index,
        metadata=metadata,
        embed_batch=embed_batch,
        corpus_records=corpus_records,
        max_deepening_steps=max_deepening_steps,
        search_web=search_web,
        read_web_page=read_web_page,
        request_cache=request_cache,
        **retrieval_config,
    )
    return EvidenceService(runtime, request_cache=request_cache)
