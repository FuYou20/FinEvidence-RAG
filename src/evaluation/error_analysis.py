import json
from pathlib import Path


def _require_non_empty_list(value, *, name):
    if not isinstance(value, list):
        raise TypeError(f"{name} 必须是列表")
    if not value:
        raise ValueError(f"{name} 不能为空")


def _require_query_id(record, *, context):
    if not isinstance(record, dict):
        raise ValueError(f"{context} 必须是字典")
    query_id = record.get("query_id")
    if not isinstance(query_id, str) or not query_id.strip():
        raise ValueError(f"{context}.query_id 必须是非空字符串")
    return query_id.strip()


def _page_key(record, *, context):
    if not isinstance(record, dict):
        raise ValueError(f"{context} 必须是字典")

    doc_id = record.get("doc_id")
    page_number = record.get("page_number")
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError(f"{context}.doc_id 必须是非空字符串")
    if (
        not isinstance(page_number, int)
        or isinstance(page_number, bool)
        or page_number <= 0
    ):
        raise ValueError(f"{context}.page_number 必须是正整数")
    return doc_id.strip(), page_number


def _evidence_id(key):
    doc_id, page_number = key
    return f"{doc_id}:p{page_number}"


def analyze_retrieval_errors(cases, run_records):
    """对齐 gold evidence 与检索页面，生成逐问题错误分析记录。

    该函数只做确定性的集合比较，不修改输入，也不调用大模型。
    """
    _require_non_empty_list(cases, name="cases")
    _require_non_empty_list(run_records, name="run_records")

    runs_by_query_id = {}
    for run_index, run_record in enumerate(run_records, start=1):
        query_id = _require_query_id(
            run_record,
            context=f"第 {run_index} 条 run_record",
        )
        if query_id in runs_by_query_id:
            raise ValueError(f"run_records 中 query_id 重复: {query_id}")
        retrieved_pages = run_record.get("retrieved_pages")
        if not isinstance(retrieved_pages, list):
            raise ValueError(
                f"run_record[{query_id}].retrieved_pages 必须是列表"
            )
        runs_by_query_id[query_id] = run_record

    case_ids = set()
    normalized_cases = []
    for case_index, case in enumerate(cases, start=1):
        query_id = _require_query_id(
            case,
            context=f"第 {case_index} 条 case",
        )
        if query_id in case_ids:
            raise ValueError(f"cases 中 query_id 重复: {query_id}")
        question = case.get("question")
        gold_evidence = case.get("gold_evidence")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"case[{query_id}].question 必须是非空字符串")
        if not isinstance(gold_evidence, list) or not gold_evidence:
            raise ValueError(
                f"case[{query_id}].gold_evidence 必须是非空列表"
            )
        case_ids.add(query_id)
        normalized_cases.append((query_id, question.strip(), gold_evidence))

    run_ids = set(runs_by_query_id)
    if case_ids != run_ids:
        missing_runs = sorted(case_ids - run_ids)
        unexpected_runs = sorted(run_ids - case_ids)
        raise ValueError(
            "cases 与 run_records 的 query_id 不一致: "
            f"missing_runs={missing_runs}, unexpected_runs={unexpected_runs}"
        )

    analyses = []
    for query_id, question, gold_evidence in normalized_cases:
        gold_pages = []
        gold_by_key = {}
        for evidence_index, evidence in enumerate(gold_evidence, start=1):
            key = _page_key(
                evidence,
                context=(
                    f"case[{query_id}].gold_evidence[{evidence_index}]"
                ),
            )
            relevance = evidence.get("relevance")
            if (
                not isinstance(relevance, int)
                or isinstance(relevance, bool)
                or relevance < 0
            ):
                raise ValueError(
                    f"case[{query_id}].gold_evidence.relevance "
                    "必须是非负整数"
                )
            if key in gold_by_key:
                existing_page = gold_by_key[key]
                existing_page["relevance"] = max(
                    existing_page["relevance"],
                    relevance,
                )
                existing_page["evidence_count"] += 1
                continue

            page = {
                "evidence_id": _evidence_id(key),
                "doc_id": key[0],
                "page_number": key[1],
                "relevance": relevance,
                "evidence_count": 1,
            }
            gold_pages.append(page)
            gold_by_key[key] = page

        retrieved_pages = []
        retrieved_by_key = {}
        raw_pages = runs_by_query_id[query_id]["retrieved_pages"]
        for page_index, page in enumerate(raw_pages, start=1):
            key = _page_key(
                page,
                context=(
                    f"run_record[{query_id}].retrieved_pages[{page_index}]"
                ),
            )
            if key in retrieved_by_key:
                raise ValueError(
                    f"run_record[{query_id}] 中检索页面重复: "
                    f"{_evidence_id(key)}"
                )
            rank = page.get("rank")
            score = page.get("score")
            if (
                not isinstance(rank, int)
                or isinstance(rank, bool)
                or rank <= 0
            ):
                raise ValueError(
                    f"run_record[{query_id}].retrieved_pages.rank "
                    "必须是正整数"
                )
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                raise ValueError(
                    f"run_record[{query_id}].retrieved_pages.score "
                    "必须是数值"
                )
            normalized_page = {
                "evidence_id": _evidence_id(key),
                "doc_id": key[0],
                "page_number": key[1],
                "rank": rank,
                "score": float(score),
            }
            retrieved_pages.append(normalized_page)
            retrieved_by_key[key] = normalized_page

        gold_keys = set(gold_by_key)
        retrieved_keys = set(retrieved_by_key)
        hit_keys = gold_keys & retrieved_keys

        relevant_hits = []
        irrelevant_retrieved_pages = []
        for page in retrieved_pages:
            key = (page["doc_id"], page["page_number"])
            if key in hit_keys:
                relevant_hits.append(
                    {**page, "relevance": gold_by_key[key]["relevance"]}
                )
            else:
                irrelevant_retrieved_pages.append(dict(page))

        missed_gold_pages = [
            dict(page)
            for page in gold_pages
            if (page["doc_id"], page["page_number"]) not in retrieved_keys
        ]
        hit_count = len(relevant_hits)
        recall = hit_count / len(gold_pages)
        if hit_count == 0:
            status = "no_recall"
            first_relevant_rank = None
        elif hit_count == len(gold_pages):
            status = "complete"
            first_relevant_rank = min(page["rank"] for page in relevant_hits)
        else:
            status = "partial_recall"
            first_relevant_rank = min(page["rank"] for page in relevant_hits)

        analyses.append(
            {
                "query_id": query_id,
                "question": question,
                "status": status,
                "gold_page_count": len(gold_pages),
                "hit_count": hit_count,
                "recall": recall,
                "first_relevant_rank": first_relevant_rank,
                "gold_pages": gold_pages,
                "retrieved_pages": retrieved_pages,
                "relevant_hits": relevant_hits,
                "missed_gold_pages": missed_gold_pages,
                "irrelevant_retrieved_pages": irrelevant_retrieved_pages,
            }
        )

    return analyses


def write_error_analysis_jsonl(output_path, analyses):
    """保存机器可读的逐问题错误分析。"""
    if not isinstance(analyses, list):
        raise TypeError("analyses 必须是列表")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in analyses:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_error_analysis_markdown(output_path, analyses):
    """保存便于人工复盘的 Markdown 错误分析报告。"""
    if not isinstance(analyses, list):
        raise TypeError("analyses 必须是列表")
    if not analyses:
        raise ValueError("analyses 不能为空")

    mean_recall = sum(record["recall"] for record in analyses) / len(analyses)
    incomplete_count = sum(
        record["status"] != "complete" for record in analyses
    )
    lines = [
        "# Retrieval Error Analysis",
        "",
        f"- Query count: {len(analyses)}",
        f"- Incomplete queries: {incomplete_count}",
        f"- Mean page recall: {mean_recall:.4f}",
        "",
        "| query_id | status | gold | hits | recall | first relevant rank |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for record in analyses:
        first_rank = record["first_relevant_rank"]
        first_rank_text = "-" if first_rank is None else str(first_rank)
        lines.append(
            f"| {record['query_id']} | {record['status']} "
            f"| {record['gold_page_count']} | {record['hit_count']} "
            f"| {record['recall']:.4f} | {first_rank_text} |"
        )

    for record in analyses:
        lines.extend(
            [
                "",
                f"## {record['query_id']} — {record['status']}",
                "",
                f"**Question:** {record['question']}",
                "",
                "**Relevant hits:**",
            ]
        )
        if record["relevant_hits"]:
            for page in record["relevant_hits"]:
                lines.append(
                    f"- {page['evidence_id']} (rank={page['rank']}, "
                    f"relevance={page['relevance']}, score={page['score']:.4f})"
                )
        else:
            lines.append("- None")

        lines.extend(["", "**Missed gold pages:**"])
        if record["missed_gold_pages"]:
            for page in record["missed_gold_pages"]:
                lines.append(
                    f"- {page['evidence_id']} "
                    f"(relevance={page['relevance']})"
                )
        else:
            lines.append("- None")

        lines.extend(["", "**Irrelevant retrieved pages:**"])
        if record["irrelevant_retrieved_pages"]:
            for page in record["irrelevant_retrieved_pages"]:
                lines.append(
                    f"- {page['evidence_id']} (rank={page['rank']}, "
                    f"score={page['score']:.4f})"
                )
        else:
            lines.append("- None")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
