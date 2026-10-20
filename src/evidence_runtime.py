from copy import deepcopy
import hashlib
import logging
from urllib.parse import urlparse

import openai

from errors import RequestError

from .chunk_expander import expand_neighbor_chunks
from .evidence_dispatcher import dispatch_evidence_request
from .evidence_cache import EvidenceCacheError
from .evidence_investigation import run_evidence_investigation
from .evidence_judge import evaluate_evidence
from .evidence_protocol import EvidenceRequest
from .evidence_registry import EvidenceRegistry
from .evidence_verification import handle_verify_evidence
from .hybrid_retriever import hybrid_retrieve
from .llm_generator import create_evidence_decision_generator
from .sparse_retriever import build_bm25_index, sparse_retrieve
from .stance_research import handle_stance_research
from .web_search import create_bocha_web_search


logger = logging.getLogger(__name__)
TRANSIENT_DENSE_ERRORS = (
    openai.APITimeoutError,
    openai.APIConnectionError,
)


class EvidenceRuntime:
    """持有真实调查依赖和证据注册表的统一运行入口。"""

    def __init__(
        self,
        *,
        registry,
        request_handler,
        request_cache=None,
        metrics=None,
    ):
        if not isinstance(registry, EvidenceRegistry):
            raise TypeError("registry 必须是 EvidenceRegistry")
        if not callable(request_handler):
            raise TypeError("request_handler 必须是可调用对象")
        if request_cache is not None:
            for method_name in ("make_key", "get", "set", "lock"):
                if not callable(getattr(request_cache, method_name, None)):
                    raise TypeError(
                        f"request_cache 缺少可调用方法: {method_name}"
                    )
        self.registry = registry
        self._request_handler = request_handler
        self._request_cache = request_cache
        self._metrics = metrics if metrics is not None else {}
        self.request_history = []

    def request_evidence(self, *, request):
        self._metrics["evidence_requests"] = (
            self._metrics.get("evidence_requests", 0) + 1
        )
        validated = EvidenceRequest.model_validate(request)
        record = {"request": validated.model_dump(mode="json")}
        self.request_history.append(record)
        cached_bundle = self._read_cache(validated, record)
        if cached_bundle is not None:
            return self._finish_cached_request(cached_bundle, record)
        if (
            self._request_cache is not None
            and record.get("cache", {}).get("status") == "miss"
        ):
            try:
                self._metrics["evidence_cache_lock_waits"] += 1
                with self._request_cache.lock(validated):
                    self._metrics["evidence_cache_lock_acquired"] += 1
                    cached_bundle = self._read_cache(
                        validated,
                        record,
                        count_miss=False,
                        hit_status="hit_after_lock",
                    )
                    if cached_bundle is not None:
                        self._metrics[
                            "evidence_cache_post_lock_hits"
                        ] += 1
                        return self._finish_cached_request(
                            cached_bundle,
                            record,
                        )
                    return self._execute_and_store(validated, record)
            except EvidenceCacheError as exc:
                self._metrics["evidence_cache_lock_errors"] += 1
                record["cache_lock"] = {
                    "status": "error",
                    "error_type": type(exc).__name__,
                }
                logger.warning(
                    "[证据缓存降级] 分布式锁失败，继续实时调查: %s",
                    exc,
                )
        return self._execute_and_store(validated, record)

    def _execute_and_store(self, request, record):
        try:
            bundle = self._request_handler(request=request)
        except Exception as exc:
            record["error_type"] = type(exc).__name__
            raise
        self._write_cache(request, bundle, record)
        record["result"] = bundle.model_dump(mode="json")
        return bundle

    def _finish_cached_request(self, bundle, record):
        self._restore_cached_evidence(bundle.evidence)
        record["result"] = bundle.model_dump(mode="json")
        return bundle

    def _read_cache(
        self,
        request,
        record,
        *,
        count_miss=True,
        hit_status="hit",
    ):
        if self._request_cache is None:
            record["cache"] = {"status": "disabled"}
            return None
        try:
            cache_key = self._request_cache.make_key(request)
            cached = self._request_cache.get(request)
        except EvidenceCacheError as exc:
            self._metrics["evidence_cache_errors"] += 1
            record["cache"] = {
                "status": "error",
                "error_type": type(exc).__name__,
            }
            logger.warning("[证据缓存降级] 读取失败，继续实时调查: %s", exc)
            return None
        if cached is None:
            if count_miss:
                self._metrics["evidence_cache_misses"] += 1
            record["cache"] = {"status": "miss", "key": cache_key}
            logger.info("[证据缓存未命中] %s", cache_key)
            return None
        bundle = cached
        self._metrics["evidence_cache_hits"] += 1
        self._metrics["avoided_investigations"] += 1
        record["cache"] = {"status": hit_status, "key": cache_key}
        logger.info("[证据缓存命中] %s", cache_key)
        return bundle

    def _write_cache(self, request, bundle, record):
        if self._request_cache is None:
            return
        try:
            cache_key = self._request_cache.set(request, bundle)
        except EvidenceCacheError as exc:
            self._metrics["evidence_cache_errors"] += 1
            record["cache_write"] = {
                "status": "error",
                "error_type": type(exc).__name__,
            }
            logger.warning("[证据缓存降级] 写入失败，不影响调查结果: %s", exc)
            return
        self._metrics["evidence_cache_writes"] += 1
        record["cache_write"] = {"status": "stored", "key": cache_key}

    def _restore_cached_evidence(self, evidence):
        for item in evidence:
            source_kind = item.get("source_kind")
            if source_kind not in {"local", "web", "multimodal", "financial"}:
                raise ValueError(
                    "缓存证据缺少有效 source_kind: "
                    f"{item.get('evidence_id')}"
                )
            self.registry.register([item], source_kind=source_kind)

    def snapshot_requests(self):
        return deepcopy(self.request_history)

    def snapshot_metrics(self):
        return deepcopy(self._metrics)


def create_evidence_runtime(
    client,
    *,
    model_name,
    dense_index,
    metadata,
    embed_batch,
    bocha_api_key=None,
    search_web=None,
    read_web_page=None,
    web_fulltext_limit=3,
    top_k=8,
    dense_k=20,
    sparse_k=20,
    neighbor_window=1,
    neighbor_seed_top_n=3,
    neighbor_decay=0.8,
    corpus_records=None,
    max_deepening_steps=2,
    request_cache=None,
):
    """组装本地 Hybrid RAG、Query Rewrite、联网搜索和证据判断。"""
    if not isinstance(metadata, list) or not metadata:
        raise ValueError("metadata 必须是非空列表")
    if not callable(embed_batch):
        raise TypeError("embed_batch 必须是可调用对象")
    for name, value in (
        ("top_k", top_k),
        ("dense_k", dense_k),
        ("sparse_k", sparse_k),
        ("neighbor_seed_top_n", neighbor_seed_top_n),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} 必须是正整数")
    if (
        not isinstance(neighbor_window, int)
        or isinstance(neighbor_window, bool)
        or neighbor_window < 0
    ):
        raise ValueError("neighbor_window 必须是非负整数")
    if (
        not isinstance(max_deepening_steps, int)
        or isinstance(max_deepening_steps, bool)
        or not 0 <= max_deepening_steps <= 4
    ):
        raise ValueError("max_deepening_steps 必须是 0 到 4 的整数")

    registry = EvidenceRegistry()
    metrics = {
        "evidence_requests": 0,
        "hybrid_retrieval_calls": 0,
        "web_search_calls": 0,
        "web_page_read_calls": 0,
        "web_page_read_successes": 0,
        "web_page_read_failures": 0,
        "search_summary_fallbacks": 0,
        "degraded_retrievals": 0,
        "dense_circuit_open": False,
        "last_dense_error_type": None,
        "evidence_cache_enabled": request_cache is not None,
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
    document_info = {
        record["doc_id"]: record
        for record in (corpus_records or [])
    }
    bm25_index = build_bm25_index(metadata)
    generate_decision = create_evidence_decision_generator(
        client,
        model_name=model_name,
    )

    raw_web_search = search_web
    if raw_web_search is None and bocha_api_key:
        raw_web_search = create_bocha_web_search(api_key=bocha_api_key)
    if raw_web_search is not None and not callable(raw_web_search):
        raise TypeError("search_web 必须是可调用对象或 None")
    if read_web_page is not None and not callable(read_web_page):
        raise TypeError("read_web_page 必须是可调用对象或 None")
    if (
        not isinstance(web_fulltext_limit, int)
        or isinstance(web_fulltext_limit, bool)
        or web_fulltext_limit < 0
    ):
        raise ValueError("web_fulltext_limit 必须是非负整数")

    dense_available = True

    def retrieve_evidence(*, query):
        nonlocal dense_available
        metrics["hybrid_retrieval_calls"] += 1
        logger.info("[Hybrid 检索] 生成查询向量并融合 Dense/BM25")
        if dense_available:
            try:
                results = hybrid_retrieve(
                    query=query,
                    dense_index=dense_index,
                    bm25_index=bm25_index,
                    metadata=metadata,
                    embed_batch=embed_batch,
                    dense_k=dense_k,
                    sparse_k=sparse_k,
                    top_k=top_k,
                )
            except TRANSIENT_DENSE_ERRORS as exc:
                dense_available = False
                metrics["dense_circuit_open"] = True
                metrics["last_dense_error_type"] = type(exc).__name__
                logger.warning(
                    "[检索降级] Dense 服务不可用，本次运行改用 BM25: %s",
                    type(exc).__name__,
                )
                results = sparse_retrieve(
                    query=query,
                    bm25_index=bm25_index,
                    metadata=metadata,
                    top_k=sparse_k,
                )
        else:
            logger.info("[检索降级] Dense 熔断已开启，直接使用 BM25")
            results = sparse_retrieve(
                query=query,
                bm25_index=bm25_index,
                metadata=metadata,
                top_k=sparse_k,
            )
        if not dense_available:
            metrics["degraded_retrievals"] += 1
            results = results[:top_k]
        if neighbor_window > 0 and results:
            results = expand_neighbor_chunks(
                results,
                metadata,
                top_k=top_k,
                seed_top_n=neighbor_seed_top_n,
                window=neighbor_window,
                decay=neighbor_decay,
            )
        enriched = []
        for result in results:
            document = document_info.get(result["doc_id"], {})
            enriched.append(
                {
                    **{
                        key: document[key]
                        for key in ("title", "published_at", "company", "document_type")
                        if key in document
                    },
                    **result,
                    "content_origin": "pdf_extraction",
                }
            )
        registered = registry.register(enriched, source_kind="local")
        logger.info("[Hybrid 检索完成] evidence=%s", len(registered))
        return registered

    registered_web_search = None
    if raw_web_search is not None:

        def registered_web_search(*, question, query, evidence):
            metrics["web_search_calls"] += 1
            logger.info("[博查搜索] 请求独立来源材料")
            results = raw_web_search(
                question=question,
                query=query,
                evidence=evidence,
            )
            if read_web_page is not None:
                enriched_results = []
                for position, result in enumerate(results):
                    if not isinstance(result, dict):
                        raise TypeError("联网搜索结果必须是 dict")
                    enriched = deepcopy(result)
                    if position >= web_fulltext_limit:
                        enriched["extraction_status"] = "not_attempted"
                        enriched_results.append(enriched)
                        continue
                    metrics["web_page_read_calls"] += 1
                    try:
                        fulltext = read_web_page(url=enriched["url"])
                    except RequestError as exc:
                        metrics["web_page_read_failures"] += 1
                        metrics["search_summary_fallbacks"] += 1
                        enriched["extraction_status"] = "failed"
                        enriched["extraction_error_type"] = type(exc).__name__
                        enriched["extraction_retryable"] = exc.retryable
                        logger.warning(
                            "[网页正文读取失败] 保留搜索摘要: %s (%s)",
                            enriched.get("url"),
                            type(exc).__name__,
                        )
                    else:
                        if not isinstance(fulltext, dict):
                            raise TypeError("read_web_page 必须返回 dict")
                        text = fulltext.get("text")
                        if not isinstance(text, str) or not text.strip():
                            raise ValueError(
                                "read_web_page 返回结果缺少非空 text"
                            )
                        summary = enriched.get("text")
                        original_url = enriched.get("url")
                        enriched.update({
                            "text": text.strip(),
                            "content_origin": "web_fulltext",
                            "extraction_status": "success",
                            "truncated": bool(
                                fulltext.get("truncated", False)
                            ),
                            "redirect_count": fulltext.get(
                                "redirect_count",
                                0,
                            ),
                        })
                        if isinstance(summary, str) and summary.strip():
                            enriched["search_summary"] = summary.strip()
                        final_url = fulltext.get("url")
                        if isinstance(final_url, str) and final_url.strip():
                            enriched["url"] = final_url.strip()
                            final_host = (
                                urlparse(final_url.strip()).netloc
                                or "web"
                            )
                            final_digest = hashlib.sha256(
                                final_url.strip().encode("utf-8")
                            ).hexdigest()[:16]
                            enriched["doc_id"] = f"web:{final_host}"
                            enriched["chunk_id"] = f"web:{final_digest}"
                            if final_url.strip() != original_url:
                                enriched["search_result_url"] = original_url
                        fulltext_title = fulltext.get("title")
                        if (
                            isinstance(fulltext_title, str)
                            and fulltext_title.strip()
                        ):
                            enriched["title"] = fulltext_title.strip()
                            enriched["source"] = fulltext_title.strip()
                        if not enriched.get("published_at"):
                            published_at = fulltext.get("published_at")
                            if (
                                isinstance(published_at, str)
                                and published_at.strip()
                            ):
                                enriched["published_at"] = (
                                    published_at.strip()
                                )
                        metrics["web_page_read_successes"] += 1
                    enriched_results.append(enriched)
                results = enriched_results
            registered = registry.register(results, source_kind="web")
            logger.info("[博查搜索完成] evidence=%s", len(registered))
            return registered

    available_actions = {"finish", "rewrite_query"}
    if registered_web_search is not None:
        available_actions.add("web_search")

    def judge_evidence(*, question, query, evidence):
        logger.info("[证据判断] 请求模型评价 %s 条证据", len(evidence))
        return evaluate_evidence(
            question=question,
            query=query,
            evidence=evidence,
            generate_decision=generate_decision,
            available_actions=available_actions,
        )

    def investigation_runner(
        *,
        question,
        max_deepening_steps,
        initial_evidence=None,
        seed_evidence=None,
    ):
        return run_evidence_investigation(
            question=question,
            retrieve_evidence=retrieve_evidence,
            judge_evidence=judge_evidence,
            initial_evidence=initial_evidence,
            seed_evidence=seed_evidence,
            rewrite_query=None,
            parse_multimodal=None,
            search_web=registered_web_search,
            max_deepening_steps=max_deepening_steps,
        )

    def request_handler(*, request):
        validated_request = EvidenceRequest.model_validate(request)
        validated_request = validated_request.model_copy(
            update={
                "max_deepening_steps": min(
                    validated_request.max_deepening_steps,
                    max_deepening_steps,
                )
            }
        )
        return dispatch_evidence_request(
            request=validated_request,
            stance_research=lambda *, request: handle_stance_research(
                request,
                investigation_runner,
                resolve_evidence=registry.get_many,
            ),
            verify_evidence=lambda *, request: handle_verify_evidence(
                request,
                investigation_runner,
                resolve_evidence=registry.get_many,
            ),
        )

    return EvidenceRuntime(
        registry=registry,
        request_handler=request_handler,
        request_cache=request_cache,
        metrics=metrics,
    )
