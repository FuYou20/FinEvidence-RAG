import json
import math
from pathlib import Path


def _gold_relevance_by_page(case):
    """同一页存在多条 quote 时，使用该页最高 relevance。"""
    relevance_by_page = {}
    for evidence in case["gold_evidence"]:
        evidence_id = f"{evidence['doc_id']}:p{evidence['page_number']}"
        relevance_by_page[evidence_id] = max(
            relevance_by_page.get(evidence_id, 0),
            evidence["relevance"],
        )
    return relevance_by_page


def _gold_relevance_by_chunk(case):
    """Build graded qrels for the exact evidence chunks."""
    relevance_by_chunk = {}
    for evidence in case["gold_evidence"]:
        chunk_id = evidence.get("source_chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(
                f"案例 {case['query_id']} 缺少 source_chunk_id，"
                "无法计算 chunk 级指标"
            )
        relevance_by_chunk[chunk_id] = max(
            relevance_by_chunk.get(chunk_id, 0),
            evidence["relevance"],
        )
    return relevance_by_chunk


def _dcg(relevances):
    return sum(
        (2**relevance - 1) / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances, start=1)
    )


def _validate_evaluation_inputs(cases, run_records, *, top_k):
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases 必须是非空列表")
    if not isinstance(run_records, list) or not run_records:
        raise ValueError("run_records 必须是非空列表")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k 必须是正整数")

    run_by_query_id = {}
    for record in run_records:
        query_id = record["query_id"]
        if query_id in run_by_query_id:
            raise ValueError(f"run_records 中 query_id 重复: {query_id}")
        run_by_query_id[query_id] = record

    case_query_ids = {case["query_id"] for case in cases}
    missing_query_ids = case_query_ids - run_by_query_id.keys()
    if missing_query_ids:
        raise ValueError(
            f"run_records 缺少 query_id: {sorted(missing_query_ids)}"
        )
    extra_query_ids = run_by_query_id.keys() - case_query_ids
    if extra_query_ids:
        raise ValueError(
            f"run_records 存在未知 query_id: {sorted(extra_query_ids)}"
        )

    return run_by_query_id


def _evaluate_ranked_retrieval(
    cases,
    run_records,
    *,
    top_k,
    gold_builder,
    prediction_field,
    prediction_id_field,
    count_field_prefix,
):
    run_by_query_id = _validate_evaluation_inputs(
        cases,
        run_records,
        top_k=top_k,
    )
    per_query = []
    for case in cases:
        query_id = case["query_id"]
        relevance_by_id = gold_builder(case)
        relevant_ids = {
            evidence_id
            for evidence_id, relevance in relevance_by_id.items()
            if relevance > 0
        }
        if not relevant_ids:
            raise ValueError(f"案例 {query_id} 没有 relevance > 0 的 gold evidence")

        predicted_ids = []
        seen_predictions = set()
        for item in run_by_query_id[query_id][prediction_field]:
            evidence_id = item[prediction_id_field]
            if evidence_id in seen_predictions:
                continue
            predicted_ids.append(evidence_id)
            seen_predictions.add(evidence_id)
            if len(predicted_ids) == top_k:
                break

        hit_ranks = [
            rank
            for rank, evidence_id in enumerate(predicted_ids, start=1)
            if evidence_id in relevant_ids
        ]
        retrieved_relevant_ids = relevant_ids.intersection(predicted_ids)
        success = 1.0 if hit_ranks else 0.0
        reciprocal_rank = 1.0 / hit_ranks[0] if hit_ranks else 0.0
        recall = len(retrieved_relevant_ids) / len(relevant_ids)

        predicted_relevances = [
            relevance_by_id.get(evidence_id, 0)
            for evidence_id in predicted_ids
        ]
        ideal_relevances = sorted(
            relevance_by_id.values(),
            reverse=True,
        )[:top_k]
        ideal_dcg = _dcg(ideal_relevances)
        ndcg = _dcg(predicted_relevances) / ideal_dcg if ideal_dcg else 0.0

        per_query.append(
            {
                "query_id": query_id,
                "success_at_k": success,
                "reciprocal_rank_at_k": reciprocal_rank,
                "recall_at_k": recall,
                "ndcg_at_k": ndcg,
                f"relevant_{count_field_prefix}_count": len(relevant_ids),
                f"retrieved_relevant_{count_field_prefix}_count": len(
                    retrieved_relevant_ids
                ),
                "first_relevant_rank": hit_ranks[0] if hit_ranks else None,
            }
        )

    query_count = len(per_query)
    return {
        "top_k": top_k,
        "query_count": query_count,
        "success_at_k": sum(item["success_at_k"] for item in per_query)
        / query_count,
        "mrr_at_k": sum(
            item["reciprocal_rank_at_k"] for item in per_query
        )
        / query_count,
        "recall_at_k": sum(item["recall_at_k"] for item in per_query)
        / query_count,
        "ndcg_at_k": sum(item["ndcg_at_k"] for item in per_query)
        / query_count,
        "per_query": per_query,
    }


def evaluate_retrieval(cases, run_records, *, top_k=5):
    """按页面级 gold 计算 Success、MRR、Recall 与 nDCG。"""
    return _evaluate_ranked_retrieval(
        cases,
        run_records,
        top_k=top_k,
        gold_builder=_gold_relevance_by_page,
        prediction_field="retrieved_pages",
        prediction_id_field="evidence_id",
        count_field_prefix="page",
    )


def evaluate_chunk_retrieval(cases, run_records, *, top_k=5):
    """按 source_chunk_id 计算精确证据片段召回指标。"""
    return _evaluate_ranked_retrieval(
        cases,
        run_records,
        top_k=top_k,
        gold_builder=_gold_relevance_by_chunk,
        prediction_field="retrieved_chunks",
        prediction_id_field="chunk_id",
        count_field_prefix="chunk",
    )


def write_metrics_report(output_path, report):
    """将指标报告写成便于复现实验的 JSON 文件。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

