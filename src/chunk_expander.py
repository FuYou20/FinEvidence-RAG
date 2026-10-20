"""Post-retrieval expansion of neighboring chunks in document order."""

import copy
import math


def _positive_int(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} 必须是正整数")


def _build_document_catalog(metadata):
    catalog = {}
    seen_chunk_ids = set()

    for row_number, item in enumerate(metadata, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"metadata 第 {row_number} 条必须是字典")
        chunk_id = item.get("chunk_id")
        doc_id = item.get("doc_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(
                f"metadata 第 {row_number} 条 chunk_id 必须是非空字符串"
            )
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError(
                f"metadata 第 {row_number} 条 doc_id 必须是非空字符串"
            )
        if chunk_id in seen_chunk_ids:
            raise ValueError(f"metadata 中 chunk_id 重复: {chunk_id}")
        seen_chunk_ids.add(chunk_id)
        catalog.setdefault(doc_id, []).append(item)

    positions = {}
    for doc_id, chunks in catalog.items():
        has_explicit_positions = all(
            isinstance(item.get("document_chunk_index"), int)
            and not isinstance(item.get("document_chunk_index"), bool)
            and item["document_chunk_index"] >= 0
            for item in chunks
        )
        if has_explicit_positions:
            chunks.sort(key=lambda item: item["document_chunk_index"])
            explicit_positions = [
                item["document_chunk_index"] for item in chunks
            ]
            if len(set(explicit_positions)) != len(explicit_positions):
                raise ValueError(
                    f"文档 {doc_id} 的 document_chunk_index 存在重复"
                )

        for position, item in enumerate(chunks):
            positions[item["chunk_id"]] = (doc_id, position)

    return catalog, positions


def _validate_retrieval_result(item, position):
    if not isinstance(item, dict):
        raise ValueError(f"retrieval_results 第 {position} 条必须是字典")
    chunk_id = item.get("chunk_id")
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise ValueError(
            f"retrieval_results 第 {position} 条 chunk_id 必须是非空字符串"
        )
    score = item.get("score")
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(float(score))
        or score < 0
    ):
        raise ValueError(
            f"retrieval_results 第 {position} 条 score 必须是非负有限数值"
        )
    return chunk_id, float(score)


def expand_neighbor_chunks(
    retrieval_results,
    metadata,
    *,
    top_k=20,
    seed_top_n=3,
    window=1,
    decay=0.8,
):
    """Add same-document neighbors and rank by distance-decayed score.

    Metadata order is the compatibility fallback for old index caches. New
    metadata may provide ``document_chunk_index`` to make the order explicit.
    Inputs are never mutated.
    """
    if not isinstance(retrieval_results, list):
        raise TypeError("retrieval_results 必须是列表")
    if not isinstance(metadata, list):
        raise TypeError("metadata 必须是列表")
    _positive_int("top_k", top_k)
    _positive_int("seed_top_n", seed_top_n)
    _positive_int("window", window)
    if (
        not isinstance(decay, (int, float))
        or isinstance(decay, bool)
        or not math.isfinite(float(decay))
        or not 0 < decay <= 1
    ):
        raise ValueError("decay 必须满足 0 < decay <= 1")
    if not retrieval_results:
        return []
    if not metadata:
        raise ValueError("metadata 不能为空")

    catalog, positions = _build_document_catalog(metadata)
    candidates = {}
    original_ranks = {}

    for result_position, item in enumerate(retrieval_results, start=1):
        chunk_id, score = _validate_retrieval_result(item, result_position)
        if chunk_id in candidates:
            raise ValueError(
                f"retrieval_results 中 chunk_id 重复: {chunk_id}"
            )
        if chunk_id not in positions:
            raise ValueError(f"检索结果 chunk_id 不存在于 metadata: {chunk_id}")

        candidate = copy.deepcopy(item)
        candidate.update(
            {
                "score": score,
                "retrieval_score": score,
                "neighbor_score": None,
                "is_direct_hit": True,
                "expanded_from": None,
                "neighbor_distance": 0,
            }
        )
        candidates[chunk_id] = candidate
        original_ranks[chunk_id] = result_position

    for seed in retrieval_results[:seed_top_n]:
        seed_id = seed["chunk_id"]
        seed_score = float(seed["score"])
        doc_id, seed_position = positions[seed_id]
        document_chunks = catalog[doc_id]

        for distance in range(1, window + 1):
            for neighbor_position in (
                seed_position - distance,
                seed_position + distance,
            ):
                if not 0 <= neighbor_position < len(document_chunks):
                    continue

                neighbor_metadata = document_chunks[neighbor_position]
                neighbor_id = neighbor_metadata["chunk_id"]
                neighbor_score = seed_score * float(decay) ** distance

                if neighbor_id not in candidates:
                    candidate = copy.deepcopy(neighbor_metadata)
                    candidate.update(
                        {
                            "score": neighbor_score,
                            "retrieval_score": None,
                            "neighbor_score": neighbor_score,
                            "is_direct_hit": False,
                            "expanded_from": seed_id,
                            "neighbor_distance": distance,
                        }
                    )
                    candidates[neighbor_id] = candidate
                    original_ranks[neighbor_id] = math.inf
                    continue

                candidate = candidates[neighbor_id]
                previous_neighbor_score = candidate["neighbor_score"]
                if (
                    previous_neighbor_score is None
                    or neighbor_score > previous_neighbor_score
                ):
                    candidate["neighbor_score"] = neighbor_score
                    candidate["expanded_from"] = seed_id
                    candidate["neighbor_distance"] = distance
                candidate["score"] = max(
                    candidate["retrieval_score"] or 0.0,
                    candidate["neighbor_score"] or 0.0,
                )

    ranked = sorted(
        candidates.values(),
        key=lambda item: (
            -item["score"],
            not item["is_direct_hit"],
            original_ranks[item["chunk_id"]],
            item["neighbor_distance"],
            item["chunk_id"],
        ),
    )[:top_k]

    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
        item["score"] = float(item["score"])
        if item["neighbor_score"] is not None:
            item["neighbor_score"] = float(item["neighbor_score"])

    return ranked
