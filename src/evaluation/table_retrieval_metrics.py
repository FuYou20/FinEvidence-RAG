import json
import math
import unicodedata
from pathlib import Path

from .retrieval_runner import load_retrieval_cases


def load_table_retrieval_cases(cases_path):
    """Load retrieval cases and validate table-specific evidence fields."""
    cases = load_retrieval_cases(cases_path)

    for case_index, case in enumerate(cases, start=1):
        answer_values = case.get("answer_values")
        if not isinstance(answer_values, list) or not answer_values:
            raise ValueError(
                f"第 {case_index} 条表格案例的 answer_values 必须是非空列表"
            )
        if any(
            not isinstance(value, str) or not value.strip()
            for value in answer_values
        ):
            raise ValueError("answer_values 中的值必须是非空字符串")

        for evidence_index, evidence in enumerate(
            case["gold_evidence"],
            start=1,
        ):
            table_id = evidence.get("table_id")
            if not isinstance(table_id, str) or not table_id.strip():
                raise ValueError(
                    f"第 {case_index} 条案例的第 {evidence_index} 条证据"
                    "缺少有效 table_id"
                )
            if evidence.get("content_type") != "table":
                raise ValueError("表格评测证据的 content_type 必须是 table")

    return cases


def _normalize_value(value):
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(normalized.split())


def _dcg(relevances):
    return sum(
        (2**relevance - 1) / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances, start=1)
    )


def _gold_relevance_by_table(case):
    relevance_by_table = {}
    for evidence in case["gold_evidence"]:
        table_id = evidence["table_id"]
        relevance_by_table[table_id] = max(
            relevance_by_table.get(table_id, 0),
            evidence["relevance"],
        )
    return relevance_by_table


def evaluate_table_retrieval(cases, run_records, *, top_k=5):
    """Evaluate table IDs and answer values in the overall chunk Top-K."""
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

    per_query = []
    for case in cases:
        query_id = case["query_id"]
        relevance_by_table = _gold_relevance_by_table(case)
        relevant_table_ids = {
            table_id
            for table_id, relevance in relevance_by_table.items()
            if relevance > 0
        }
        if not relevant_table_ids:
            raise ValueError(f"案例 {query_id} 没有 relevance > 0 的表格证据")

        chunks = run_by_query_id[query_id].get("retrieved_chunks")
        if not isinstance(chunks, list):
            raise ValueError(f"案例 {query_id} 缺少 retrieved_chunks 列表")

        seen_table_ids = set()
        retrieved_relevant_ids = set()
        hit_ranks = []
        predicted_relevances = []
        relevant_chunk_texts = []

        for rank, chunk in enumerate(chunks[:top_k], start=1):
            if not isinstance(chunk, dict):
                raise ValueError(f"案例 {query_id} 第 {rank} 条结果必须是字典")

            table_id = chunk.get("table_id")
            is_table = chunk.get("content_type") == "table"
            if is_table and table_id in relevant_table_ids:
                relevant_chunk_texts.append(chunk.get("text", ""))

            relevance = 0
            if (
                is_table
                and isinstance(table_id, str)
                and table_id not in seen_table_ids
            ):
                seen_table_ids.add(table_id)
                relevance = relevance_by_table.get(table_id, 0)
                if relevance > 0:
                    hit_ranks.append(rank)
                    retrieved_relevant_ids.add(table_id)
            predicted_relevances.append(relevance)

        success = 1.0 if hit_ranks else 0.0
        reciprocal_rank = 1.0 / hit_ranks[0] if hit_ranks else 0.0
        recall = len(retrieved_relevant_ids) / len(relevant_table_ids)
        ideal_relevances = sorted(
            relevance_by_table.values(),
            reverse=True,
        )[:top_k]
        ideal_dcg = _dcg(ideal_relevances)
        ndcg = _dcg(predicted_relevances) / ideal_dcg if ideal_dcg else 0.0

        normalized_text = _normalize_value("\n".join(relevant_chunk_texts))
        answer_values = case["answer_values"]
        matched_answer_values = [
            value
            for value in answer_values
            if _normalize_value(value) in normalized_text
        ]
        answer_value_recall = len(matched_answer_values) / len(answer_values)
        answer_value_hit = 1.0 if answer_value_recall == 1.0 else 0.0

        per_query.append(
            {
                "query_id": query_id,
                "table_success_at_k": success,
                "table_reciprocal_rank_at_k": reciprocal_rank,
                "table_recall_at_k": recall,
                "table_ndcg_at_k": ndcg,
                "answer_value_hit_at_k": answer_value_hit,
                "answer_value_recall_at_k": answer_value_recall,
                "relevant_table_count": len(relevant_table_ids),
                "retrieved_relevant_table_count": len(
                    retrieved_relevant_ids
                ),
                "first_relevant_rank": hit_ranks[0] if hit_ranks else None,
                "matched_answer_values": matched_answer_values,
            }
        )

    query_count = len(per_query)
    return {
        "top_k": top_k,
        "query_count": query_count,
        "table_success_at_k": sum(
            item["table_success_at_k"] for item in per_query
        )
        / query_count,
        "table_mrr_at_k": sum(
            item["table_reciprocal_rank_at_k"] for item in per_query
        )
        / query_count,
        "table_recall_at_k": sum(
            item["table_recall_at_k"] for item in per_query
        )
        / query_count,
        "table_ndcg_at_k": sum(
            item["table_ndcg_at_k"] for item in per_query
        )
        / query_count,
        "answer_value_hit_at_k": sum(
            item["answer_value_hit_at_k"] for item in per_query
        )
        / query_count,
        "answer_value_recall_at_k": sum(
            item["answer_value_recall_at_k"] for item in per_query
        )
        / query_count,
        "per_query": per_query,
    }


def write_table_metrics_report(output_path, report):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
