from collections import Counter, defaultdict


RETRIEVAL_CATEGORIES = frozenset(
    {"direct_text", "boundary", "table", "text_table", "multi_section"}
)


def _source_ids(case):
    return [item.get("source_chunk_id") for item in case["gold_evidence"]]


def _validate_category(case, sources):
    category = case.get("category")
    query_id = case["query_id"]
    if category not in RETRIEVAL_CATEGORIES:
        raise ValueError(f"{query_id} 的 category 不受支持: {category}")

    content_types = [source.get("content_type", "text") for source in sources]
    if category == "direct_text" and "text" not in content_types:
        raise ValueError(f"{query_id} 的 direct_text 案例缺少正文证据")
    if category == "table" and "table" not in content_types:
        raise ValueError(f"{query_id} 的 table 案例缺少表格证据")
    if category == "text_table" and not {"text", "table"}.issubset(
        content_types
    ):
        raise ValueError(f"{query_id} 的 text_table 案例必须同时包含正文和表格")
    if category == "multi_section":
        text_sources = [
            source
            for source in sources
            if source.get("content_type", "text") == "text"
        ]
        pages = {source["page_number"] for source in text_sources}
        if len(text_sources) < 2 or len(pages) < 2:
            raise ValueError(
                f"{query_id} 的 multi_section 案例必须包含不同页面的正文证据"
            )
    if category == "boundary":
        if len(sources) < 2 or any(
            source.get("content_type", "text") != "text" for source in sources
        ):
            raise ValueError(f"{query_id} 的 boundary 案例必须包含相邻正文证据")
        document_indices = sorted(
            source.get("document_chunk_index") for source in sources
        )
        if any(not isinstance(index, int) for index in document_indices):
            raise ValueError(f"{query_id} 的 boundary 证据缺少文档内序号")
        if any(
            right - left != 1
            for left, right in zip(document_indices, document_indices[1:])
        ):
            raise ValueError(f"{query_id} 的 boundary 证据在文档中并不相邻")


def validate_retrieval_dataset(
    cases,
    metadata,
    *,
    expected_total=None,
    expected_category_count=None,
    require_disjoint_splits=False,
):
    """验证检索评测集与索引 metadata 的可追溯性及类别约束。"""
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases 必须是非空列表")
    if not isinstance(metadata, list) or not metadata:
        raise ValueError("metadata 必须是非空列表")
    if expected_total is not None and len(cases) != expected_total:
        raise ValueError(f"评测案例数量应为 {expected_total}，实际为 {len(cases)}")

    source_by_id = {item["chunk_id"]: item for item in metadata}
    category_counts = Counter()
    split_counts = Counter()
    split_documents = defaultdict(set)
    seen_query_ids = set()

    for case in cases:
        query_id = case.get("query_id")
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError("query_id 必须是非空字符串")
        if query_id in seen_query_ids:
            raise ValueError(f"query_id 重复: {query_id}")
        seen_query_ids.add(query_id)

        evidence = case.get("gold_evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"{query_id} 缺少 gold_evidence")

        sources = []
        for item in evidence:
            source_id = item.get("source_chunk_id")
            if source_id not in source_by_id:
                raise ValueError(f"{query_id} 引用了不存在的 source_chunk_id: {source_id}")
            source = source_by_id[source_id]
            for field, expected in (
                ("doc_id", source["doc_id"]),
                ("page_number", source["page_number"]),
                ("evidence_type", source.get("content_type", "text")),
                ("quote", source["text"]),
            ):
                if item.get(field) != expected:
                    raise ValueError(
                        f"{query_id} 的 {source_id} 字段 {field} 与索引 metadata 不一致"
                    )
            sources.append(source)

        _validate_category(case, sources)
        category_counts[case["category"]] += 1
        split = case.get("split")
        if split not in {"dev", "test"}:
            raise ValueError(f"{query_id} 的 split 必须是 dev 或 test")
        split_counts[split] += 1
        split_documents[split].update(source["doc_id"] for source in sources)

    if expected_category_count is not None:
        unexpected = {
            category: category_counts[category]
            for category in RETRIEVAL_CATEGORIES
            if category_counts[category] != expected_category_count
        }
        if unexpected:
            raise ValueError(
                "各类别案例数量不符合预期: "
                f"expected={expected_category_count}, actual={unexpected}"
            )

    overlap = split_documents["dev"] & split_documents["test"]
    if require_disjoint_splits and overlap:
        raise ValueError(f"dev/test 存在文档泄漏: {sorted(overlap)}")

    return {
        "total": len(cases),
        "category_counts": dict(sorted(category_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "split_documents": {
            split: sorted(documents)
            for split, documents in sorted(split_documents.items())
        },
        "split_document_overlap": sorted(overlap),
    }
