import json
from pathlib import Path
from time import perf_counter

from ..retriever import retrieve


REQUIRED_CASE_FIELDS = frozenset({"query_id", "question", "gold_evidence"})
REQUIRED_RESULT_FIELDS = frozenset(
    {"doc_id", "page_number", "chunk_id", "score"}
)


def load_retrieval_cases(cases_path):
    """读取并验证 JSONL 格式的可回答检索评测集。"""
    cases_path = Path(cases_path)
    cases = []
    seen_query_ids = set()

    with cases_path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            stripped_line = raw_line.strip()
            if not stripped_line:
                continue

            try:
                case = json.loads(stripped_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"检索评测集 {cases_path} 第 {line_number} 行不是合法 JSON"
                ) from exc

            if not isinstance(case, dict):
                raise ValueError(
                    f"检索评测集 {cases_path} 第 {line_number} 行必须是 JSON 对象"
                )

            missing_fields = REQUIRED_CASE_FIELDS - case.keys()
            if missing_fields:
                raise ValueError(
                    f"检索评测集 {cases_path} 第 {line_number} 行缺少必填字段: "
                    f"{sorted(missing_fields)}"
                )

            query_id = case["query_id"]
            question = case["question"]
            gold_evidence = case["gold_evidence"]

            if not isinstance(query_id, str) or not query_id.strip():
                raise ValueError(
                    f"检索评测集 {cases_path} 第 {line_number} 行的 "
                    "query_id 必须是非空字符串"
                )
            normalized_query_id = query_id.strip()
            if normalized_query_id in seen_query_ids:
                raise ValueError(
                    f"检索评测集 {cases_path} 第 {line_number} 行的 "
                    f"query_id 重复: {normalized_query_id}"
                )
            if not isinstance(question, str) or not question.strip():
                raise ValueError(
                    f"检索评测集 {cases_path} 第 {line_number} 行的 "
                    "question 必须是非空字符串"
                )
            if not isinstance(gold_evidence, list) or not gold_evidence:
                raise ValueError(
                    f"检索评测集 {cases_path} 第 {line_number} 行的 "
                    "gold_evidence 必须是非空列表"
                )

            for evidence_index, evidence in enumerate(gold_evidence, start=1):
                if not isinstance(evidence, dict):
                    raise ValueError(
                        f"检索评测集 {cases_path} 第 {line_number} 行的第 "
                        f"{evidence_index} 条 gold_evidence 必须是 JSON 对象"
                    )
                for field in ("doc_id", "page_number", "relevance"):
                    if field not in evidence:
                        raise ValueError(
                            f"检索评测集 {cases_path} 第 {line_number} 行的第 "
                            f"{evidence_index} 条 gold_evidence 缺少字段: {field}"
                        )
                if not isinstance(evidence["doc_id"], str) or not evidence[
                    "doc_id"
                ].strip():
                    raise ValueError("gold_evidence.doc_id 必须是非空字符串")
                if (
                    not isinstance(evidence["page_number"], int)
                    or isinstance(evidence["page_number"], bool)
                    or evidence["page_number"] <= 0
                ):
                    raise ValueError("gold_evidence.page_number 必须是正整数")
                if (
                    not isinstance(evidence["relevance"], int)
                    or isinstance(evidence["relevance"], bool)
                    or evidence["relevance"] < 0
                ):
                    raise ValueError("gold_evidence.relevance 必须是非负整数")

            normalized_case = dict(case)
            normalized_case["query_id"] = normalized_query_id
            normalized_case["question"] = question.strip()
            seen_query_ids.add(normalized_query_id)
            cases.append(normalized_case)

    return cases


def collapse_chunks_to_pages(retrieval_results, *, top_k):
    """按原始排名将 chunk 结果折叠为唯一页面结果。"""
    if not isinstance(retrieval_results, list):
        raise TypeError("retrieval_results 必须是列表")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k 必须是正整数")

    pages = []
    seen_pages = set()

    for result_index, result in enumerate(retrieval_results, start=1):
        if not isinstance(result, dict):
            raise ValueError(f"第 {result_index} 条检索结果必须是字典")
        missing_fields = REQUIRED_RESULT_FIELDS - result.keys()
        if missing_fields:
            raise ValueError(
                f"第 {result_index} 条检索结果缺少字段: {sorted(missing_fields)}"
            )

        doc_id = result["doc_id"]
        page_number = result["page_number"]
        page_key = (doc_id, page_number)
        if page_key in seen_pages:
            continue

        pages.append(
            {
                "evidence_id": f"{doc_id}:p{page_number}",
                "doc_id": doc_id,
                "page_number": page_number,
                "rank": len(pages) + 1,
                "score": float(result["score"]),
                "best_chunk_id": result["chunk_id"],
                "best_content_type": result.get("content_type", "text"),
            }
        )
        seen_pages.add(page_key)

        if len(pages) == top_k:
            break

    return pages


def run_retrieval_cases_with_retriever(
    cases,
    retrieve_fn,
    *,
    candidate_k=20,
    page_top_k=5,
):
    """使用注入的检索函数运行案例，返回不包含 gold 标签的预测记录。"""
    if not isinstance(cases, list):
        raise TypeError("cases 必须是列表")
    if not cases:
        raise ValueError("cases 不能为空")
    if not callable(retrieve_fn):
        raise TypeError("retrieve_fn 必须是可调用对象")
    for name, value in (
        ("candidate_k", candidate_k),
        ("page_top_k", page_top_k),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} 必须是正整数")
    if candidate_k < page_top_k:
        raise ValueError("candidate_k 不能小于 page_top_k")

    run_records = []
    seen_query_ids = set()

    for case in cases:
        query_id = case["query_id"]
        question = case["question"]
        if query_id in seen_query_ids:
            raise ValueError(f"query_id 重复: {query_id}")

        started_at = perf_counter()
        chunk_results = retrieve_fn(
            query=question,
            top_k=candidate_k,
        )
        latency_ms = (perf_counter() - started_at) * 1000
        retrieved_pages = collapse_chunks_to_pages(
            chunk_results,
            top_k=page_top_k,
        )
        retrieved_chunks = [
            {
                "chunk_id": item["chunk_id"],
                "doc_id": item["doc_id"],
                "page_number": item["page_number"],
                "rank": item.get("rank", position),
                "score": float(item["score"]),
                "content_type": item.get("content_type", "text"),
                "table_id": item.get("table_id"),
                "text": item.get("text", ""),
                "is_direct_hit": item.get("is_direct_hit", True),
                "expanded_from": item.get("expanded_from"),
                "neighbor_distance": item.get("neighbor_distance", 0),
                **(
                    {
                        "retrieval_rank": item["retrieval_rank"],
                        "retrieval_score": float(item["retrieval_score"]),
                        "rerank_score": float(item["rerank_score"]),
                    }
                    if all(
                        field in item
                        for field in (
                            "retrieval_rank",
                            "retrieval_score",
                            "rerank_score",
                        )
                    )
                    else {}
                ),
            }
            for position, item in enumerate(chunk_results, start=1)
        ]
        expanded_neighbor_count = sum(
            item["is_direct_hit"] is False
            for item in retrieved_chunks
        )
        boosted_direct_count = sum(
            item.get("is_direct_hit") is True
            and item.get("neighbor_score") is not None
            for item in chunk_results
        )

        run_records.append(
            {
                "query_id": query_id,
                "question": question,
                "candidate_k": candidate_k,
                "page_top_k": page_top_k,
                "retrieved_chunk_count": len(chunk_results),
                "expanded_neighbor_count": expanded_neighbor_count,
                "boosted_direct_count": boosted_direct_count,
                "retrieved_page_count": len(retrieved_pages),
                "latency_ms": latency_ms,
                "retrieved_chunks": retrieved_chunks,
                "retrieved_pages": retrieved_pages,
            }
        )
        seen_query_ids.add(query_id)

    return run_records


def run_retrieval_cases(
    cases,
    index,
    metadata,
    embed_batch,
    *,
    candidate_k=20,
    page_top_k=5,
):
    """使用 Dense Retrieval 运行全部检索案例的兼容入口。"""

    def retrieve_fn(*, query, top_k):
        return retrieve(
            query=query,
            index=index,
            metadata=metadata,
            embed_batch=embed_batch,
            top_k=top_k,
        )

    return run_retrieval_cases_with_retriever(
        cases,
        retrieve_fn,
        candidate_k=candidate_k,
        page_top_k=page_top_k,
    )


def write_retrieval_run(output_path, run_records):
    """将检索预测写成一行一条记录的 UTF-8 JSONL 文件。"""
    if not isinstance(run_records, list):
        raise TypeError("run_records 必须是列表")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in run_records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

