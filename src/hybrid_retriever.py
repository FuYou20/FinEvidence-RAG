import copy

from .retriever import retrieve as dense_retrieve
from .sparse_retriever import sparse_retrieve, tokenize_text


def _validate_ranked_result(item, *, source, position):
    if not isinstance(item, dict):
        raise ValueError(f"{source} 第 {position} 条结果必须是字典")
    chunk_id = item.get("chunk_id")
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise ValueError(
            f"{source} 第 {position} 条结果的 chunk_id 必须是非空字符串"
        )
    rank = item.get("rank")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise ValueError(
            f"{source} 第 {position} 条结果的 rank 必须是正整数"
        )
    score = item.get("score")
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        raise ValueError(
            f"{source} 第 {position} 条结果的 score 必须是数值"
        )
    return chunk_id.strip(), rank, float(score)


def reciprocal_rank_fusion(
    dense_results,
    sparse_results,
    *,
    top_k=20,
    rrf_k=60,
    dense_weight=1.0,
    sparse_weight=1.0,
):
    """按 chunk_id 使用加权 Reciprocal Rank Fusion 融合两路排名。"""
    if not isinstance(dense_results, list):
        raise TypeError("dense_results 必须是列表")
    if not isinstance(sparse_results, list):
        raise TypeError("sparse_results 必须是列表")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k 必须是正整数")
    if not isinstance(rrf_k, int) or isinstance(rrf_k, bool) or rrf_k <= 0:
        raise ValueError("rrf_k 必须是正整数")
    for name, weight in (
        ("dense_weight", dense_weight),
        ("sparse_weight", sparse_weight),
    ):
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            raise TypeError(f"{name} 必须是数值")
        if weight < 0:
            raise ValueError(f"{name} 不能为负数")
    if dense_weight == 0 and sparse_weight == 0:
        raise ValueError("dense_weight 和 sparse_weight 不能同时为 0")

    merged = {}

    def add_results(results, *, source, weight):
        rank_field = f"{source}_rank"
        score_field = f"{source}_score"
        source_label = "bm25" if source == "sparse" else "dense"
        seen_chunk_ids = set()

        for position, item in enumerate(results, start=1):
            chunk_id, rank, original_score = _validate_ranked_result(
                item,
                source=source,
                position=position,
            )
            if chunk_id in seen_chunk_ids:
                raise ValueError(f"{source}_results 中 chunk_id 重复: {chunk_id}")
            seen_chunk_ids.add(chunk_id)

            if chunk_id not in merged:
                base = copy.deepcopy(item)
                base.update(
                    {
                        "dense_rank": None,
                        "dense_score": None,
                        "sparse_rank": None,
                        "sparse_score": None,
                        "rrf_score": 0.0,
                        "retrieval_source": [],
                    }
                )
                merged[chunk_id] = base

            fused_item = merged[chunk_id]
            fused_item[rank_field] = rank
            fused_item[score_field] = original_score
            fused_item["rrf_score"] += weight / (rrf_k + rank)
            fused_item["retrieval_source"].append(source_label)

    add_results(dense_results, source="dense", weight=dense_weight)
    add_results(sparse_results, source="sparse", weight=sparse_weight)

    fused_results = sorted(
        merged.values(),
        key=lambda item: (
            -item["rrf_score"],
            min(
                rank
                for rank in (item["dense_rank"], item["sparse_rank"])
                if rank is not None
            ),
            item["chunk_id"],
        ),
    )[:top_k]

    for final_rank, item in enumerate(fused_results, start=1):
        item["rank"] = final_rank
        item["rrf_score"] = float(item["rrf_score"])
        item["score"] = item["rrf_score"]

    return fused_results


def hybrid_retrieve(
    query,
    dense_index,
    bm25_index,
    metadata,
    embed_batch,
    *,
    dense_k=20,
    sparse_k=20,
    top_k=20,
    rrf_k=60,
    dense_weight=1.0,
    sparse_weight=1.0,
    tokenizer=tokenize_text,
):
    """分别执行 Dense、BM25 检索，再使用 RRF 返回统一结果。"""
    dense_results = dense_retrieve(
        query=query,
        index=dense_index,
        metadata=metadata,
        embed_batch=embed_batch,
        top_k=dense_k,
    )
    sparse_results = sparse_retrieve(
        query=query,
        bm25_index=bm25_index,
        metadata=metadata,
        top_k=sparse_k,
        tokenizer=tokenizer,
    )
    return reciprocal_rank_fusion(
        dense_results,
        sparse_results,
        top_k=top_k,
        rrf_k=rrf_k,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
    )
