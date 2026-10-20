import copy
import math

import requests

from errors import RequestError


DEFAULT_DASHSCOPE_RERANK_ENDPOINT = (
    "https://dashscope.aliyuncs.com/api/v1/services/"
    "rerank/text-rerank/text-rerank"
)
DEFAULT_RERANK_INSTRUCTION = (
    "Given a financial research question, retrieve passages that provide "
    "direct, specific, and verifiable evidence for answering the question."
)


def _normalize_required_string(value, *, field_name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} 必须是非空字符串")
    return value.strip()


def _validate_positive_int(value, *, field_name):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} 必须是正整数")


def create_dashscope_rerank_batch(
    *,
    api_key,
    model_name="qwen3.7-text-rerank",
    endpoint=DEFAULT_DASHSCOPE_RERANK_ENDPOINT,
    instruction=DEFAULT_RERANK_INSTRUCTION,
    timeout=45.0,
    http_client=requests,
):
    """创建一个 Query-Document 联合打分的百炼文本重排函数。

    返回函数遵守 ``rerank_batch(query, documents, top_n=...)`` 协议，
    因而可以在不改动检索主链路的情况下替换为本地 CrossEncoder。
    """
    normalized_api_key = _normalize_required_string(
        api_key,
        field_name="api_key",
    )
    normalized_model_name = _normalize_required_string(
        model_name,
        field_name="model_name",
    )
    normalized_endpoint = _normalize_required_string(
        endpoint,
        field_name="endpoint",
    )
    normalized_instruction = _normalize_required_string(
        instruction,
        field_name="instruction",
    )
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
        raise TypeError("timeout 必须是数值")
    if timeout <= 0:
        raise ValueError("timeout 必须是正数")
    post = getattr(http_client, "post", None)
    if not callable(post):
        raise TypeError("http_client 必须提供可调用的 post()")

    def rerank_batch(query, documents, *, top_n):
        normalized_query = _normalize_required_string(
            query,
            field_name="query",
        )
        if not isinstance(documents, list) or not documents:
            raise ValueError("documents 必须是非空列表")
        normalized_documents = [
            _normalize_required_string(document, field_name=f"documents[{index}]")
            for index, document in enumerate(documents)
        ]
        _validate_positive_int(top_n, field_name="top_n")
        effective_top_n = min(top_n, len(normalized_documents))

        try:
            response = post(
                normalized_endpoint,
                headers={
                    "Authorization": f"Bearer {normalized_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": normalized_model_name,
                    "input": {
                        "query": normalized_query,
                        "documents": normalized_documents,
                    },
                    "parameters": {
                        "top_n": effective_top_n,
                        "instruct": normalized_instruction,
                    },
                },
                timeout=float(timeout),
            )
            response.raise_for_status()
            try:
                payload = response.json()
                if not isinstance(payload, dict):
                    raise TypeError("response JSON must be an object")
                provider_code = payload.get("code")
                if provider_code not in (None, 200, "200"):
                    numeric_code = (
                        int(provider_code)
                        if str(provider_code).isdigit()
                        else None
                    )
                    raise RequestError(
                        f"重排服务业务错误: {provider_code}",
                        retryable=numeric_code == 429
                        or (numeric_code is not None and numeric_code >= 500),
                        status_code=numeric_code,
                    )
                results = payload["output"]["results"]
            except (KeyError, TypeError, ValueError) as exc:
                raise RequestError(
                    "重排服务返回结构不符合预期",
                    retryable=True,
                    status_code=getattr(response, "status_code", None),
                ) from exc
            if not isinstance(results, list):
                raise RequestError(
                    "重排服务 output.results 必须是列表",
                    retryable=True,
                    status_code=getattr(response, "status_code", None),
                )
            return results
        except requests.exceptions.Timeout as exc:
            raise RequestError(
                "重排服务请求超时",
                retryable=True,
                status_code=408,
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise RequestError(
                "重排服务连接失败",
                retryable=True,
                status_code=502,
            ) from exc
        except requests.exceptions.HTTPError as exc:
            status_code = getattr(exc.response, "status_code", None)
            retryable = status_code == 429 or (
                status_code is not None and status_code >= 500
            )
            raise RequestError(
                f"重排服务 HTTP 错误: {status_code}",
                retryable=retryable,
                status_code=status_code,
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RequestError(
                "重排服务请求失败",
                retryable=True,
            ) from exc

    return rerank_batch


def rerank_results(
    query,
    retrieval_results,
    rerank_batch,
    *,
    top_k=5,
):
    """使用联合相关性分数对一组召回候选重新排序。

    原始候选不会被修改。返回结果保留 ``retrieval_rank`` 和
    ``retrieval_score``，并以 ``rerank_score`` 作为新的公共 ``score``。
    """
    normalized_query = _normalize_required_string(query, field_name="query")
    if not isinstance(retrieval_results, list):
        raise TypeError("retrieval_results 必须是列表")
    if not retrieval_results:
        return []
    if not callable(rerank_batch):
        raise TypeError("rerank_batch 必须是可调用对象")
    _validate_positive_int(top_k, field_name="top_k")

    documents = []
    candidates = []
    for position, item in enumerate(retrieval_results, start=1):
        if not isinstance(item, dict):
            raise TypeError(f"第 {position} 条检索结果必须是字典")
        text = _normalize_required_string(
            item.get("text"),
            field_name=f"retrieval_results[{position - 1}].text",
        )
        score = item.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError(f"第 {position} 条检索结果的 score 必须是数值")
        if not math.isfinite(float(score)):
            raise ValueError(f"第 {position} 条检索结果的 score 必须是有限数")
        rank = item.get("rank", position)
        if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            raise ValueError(f"第 {position} 条检索结果的 rank 必须是正整数")
        documents.append(text)
        candidates.append(copy.deepcopy(item))

    effective_top_k = min(top_k, len(candidates))
    scored_results = rerank_batch(
        normalized_query,
        documents,
        top_n=effective_top_k,
    )
    if not isinstance(scored_results, list) or not scored_results:
        raise ValueError("rerank_batch 必须返回非空列表")
    if len(scored_results) > effective_top_k:
        raise ValueError("rerank_batch 返回结果数超过 top_k")

    reranked = []
    seen_indices = set()
    for new_rank, scored_item in enumerate(scored_results, start=1):
        if not isinstance(scored_item, dict):
            raise ValueError("rerank_batch 的每条结果必须是字典")
        candidate_index = scored_item.get("index")
        rerank_score = scored_item.get("relevance_score")
        if (
            not isinstance(candidate_index, int)
            or isinstance(candidate_index, bool)
            or not 0 <= candidate_index < len(candidates)
        ):
            raise ValueError(f"无效的重排候选下标: {candidate_index}")
        if candidate_index in seen_indices:
            raise ValueError(f"重排候选下标重复: {candidate_index}")
        if (
            not isinstance(rerank_score, (int, float))
            or isinstance(rerank_score, bool)
            or not math.isfinite(float(rerank_score))
        ):
            raise ValueError("relevance_score 必须是有限数值")

        result = candidates[candidate_index]
        result.update(
            {
                "retrieval_rank": result.get("rank", candidate_index + 1),
                "retrieval_score": float(result["score"]),
                "rerank_score": float(rerank_score),
                "rank": new_rank,
                "score": float(rerank_score),
            }
        )
        reranked.append(result)
        seen_indices.add(candidate_index)

    return reranked
