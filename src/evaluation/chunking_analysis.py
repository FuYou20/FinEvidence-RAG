"""Utilities for comparing fixed-window chunks with Docling chunks."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, median


SENTENCE_ENDINGS = set("。！？；.!?;：:)]）】}\"'”’")


def load_docling_chunks(path):
    """Load and minimally validate Docling's JSONL chunk export."""
    path = Path(path)
    chunks = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Docling JSONL 第 {line_number} 行不是合法 JSON"
                ) from exc

            if not isinstance(chunk, dict):
                raise ValueError(
                    f"Docling JSONL 第 {line_number} 行必须是对象"
                )
            missing = {"chunk_index", "text", "num_tokens"} - chunk.keys()
            if missing:
                raise ValueError(
                    f"Docling JSONL 第 {line_number} 行缺少字段: "
                    f"{sorted(missing)}"
                )
            if not isinstance(chunk["text"], str):
                raise ValueError(
                    f"Docling JSONL 第 {line_number} 行 text 必须是字符串"
                )
            chunks.append(chunk)

    if not chunks:
        raise ValueError("Docling JSONL 中没有 chunk")
    return chunks


def _percentile(values, fraction):
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def _length_summary(values):
    if not values:
        return {
            "min": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0,
            "max": 0,
        }
    return {
        "min": min(values),
        "mean": round(mean(values), 2),
        "median": round(median(values), 2),
        "p90": _percentile(values, 0.9),
        "max": max(values),
    }


def _looks_mid_sentence(text):
    normalized = text.rstrip()
    return bool(normalized) and normalized[-1] not in SENTENCE_ENDINGS


def analyze_fixed_chunks(chunks, *, source_character_count):
    """Summarize the project's current page-local fixed-window chunks."""
    if not chunks:
        raise ValueError("fixed chunks 不能为空")
    if source_character_count <= 0:
        raise ValueError("source_character_count 必须大于 0")

    character_lengths = [len(chunk["text"]) for chunk in chunks]
    total_chunk_characters = sum(character_lengths)
    mid_sentence_count = sum(
        _looks_mid_sentence(chunk["text"]) for chunk in chunks
    )

    return {
        "strategy": "fixed_window",
        "chunk_count": len(chunks),
        "character_lengths": _length_summary(character_lengths),
        "under_64_characters": sum(length < 64 for length in character_lengths),
        "under_120_characters": sum(length < 120 for length in character_lengths),
        "mid_sentence_end_count_heuristic": mid_sentence_count,
        "mid_sentence_end_rate_heuristic": round(
            mid_sentence_count / len(chunks), 4
        ),
        "multi_page_chunk_count": 0,
        "source_character_count": source_character_count,
        "total_chunk_character_count": total_chunk_characters,
        "overlap_character_overhead_ratio": round(
            total_chunk_characters / source_character_count,
            4,
        ),
        "heading_metadata_chunk_count": None,
        "table_aware_chunk_count": None,
        "picture_aware_chunk_count": None,
    }


def analyze_docling_chunks(chunks, *, token_limit=512):
    """Summarize structural metadata and size distribution in Docling chunks."""
    if not chunks:
        raise ValueError("Docling chunks 不能为空")
    if token_limit <= 0:
        raise ValueError("token_limit 必须大于 0")

    character_lengths = [len(chunk["text"]) for chunk in chunks]
    token_lengths = [int(chunk["num_tokens"]) for chunk in chunks]

    def doc_items(chunk):
        value = chunk.get("doc_items") or []
        return value if isinstance(value, list) else []

    def page_numbers(chunk):
        value = chunk.get("page_numbers") or []
        return value if isinstance(value, list) else []

    table_indices = [
        chunk["chunk_index"]
        for chunk in chunks
        if any("/tables/" in item for item in doc_items(chunk))
    ]
    picture_indices = [
        chunk["chunk_index"]
        for chunk in chunks
        if any("/pictures/" in item for item in doc_items(chunk))
    ]
    multi_page_indices = [
        chunk["chunk_index"]
        for chunk in chunks
        if len(set(page_numbers(chunk))) > 1
    ]
    heading_indices = [
        chunk["chunk_index"]
        for chunk in chunks
        if chunk.get("headings")
    ]
    mid_sentence_count = sum(
        _looks_mid_sentence(chunk["text"]) for chunk in chunks
    )
    covered_pages = sorted(
        {
            page_number
            for chunk in chunks
            for page_number in page_numbers(chunk)
            if isinstance(page_number, int)
        }
    )

    return {
        "strategy": "docling_hybrid",
        "chunk_count": len(chunks),
        "character_lengths": _length_summary(character_lengths),
        "token_lengths": _length_summary(token_lengths),
        "under_64_characters": sum(length < 64 for length in character_lengths),
        "under_120_characters": sum(length < 120 for length in character_lengths),
        "under_64_tokens": sum(length < 64 for length in token_lengths),
        "over_token_limit_count": sum(
            length > token_limit for length in token_lengths
        ),
        "mid_sentence_end_count_heuristic": mid_sentence_count,
        "mid_sentence_end_rate_heuristic": round(
            mid_sentence_count / len(chunks), 4
        ),
        "multi_page_chunk_count": len(multi_page_indices),
        "multi_page_chunk_indices": multi_page_indices,
        "heading_metadata_chunk_count": len(heading_indices),
        "heading_metadata_rate": round(len(heading_indices) / len(chunks), 4),
        "table_aware_chunk_count": len(table_indices),
        "table_chunk_indices": table_indices,
        "picture_aware_chunk_count": len(picture_indices),
        "picture_chunk_indices": picture_indices,
        "covered_page_count": len(covered_pages),
        "covered_pages": covered_pages,
    }


def build_comparison_report(fixed_metrics, docling_metrics):
    """Render a compact Markdown comparison report."""
    fixed_lengths = fixed_metrics["character_lengths"]
    docling_lengths = docling_metrics["character_lengths"]
    docling_tokens = docling_metrics["token_lengths"]

    rows = [
        ("Chunk 数量", fixed_metrics["chunk_count"], docling_metrics["chunk_count"]),
        ("平均字符数", fixed_lengths["mean"], docling_lengths["mean"]),
        ("字符数中位数", fixed_lengths["median"], docling_lengths["median"]),
        ("字符数 P90", fixed_lengths["p90"], docling_lengths["p90"]),
        ("最大字符数", fixed_lengths["max"], docling_lengths["max"]),
        ("少于 120 字符", fixed_metrics["under_120_characters"], docling_metrics["under_120_characters"]),
        ("疑似句中截断（启发式）", fixed_metrics["mid_sentence_end_count_heuristic"], docling_metrics["mid_sentence_end_count_heuristic"]),
        ("跨页 Chunk", fixed_metrics["multi_page_chunk_count"], docling_metrics["multi_page_chunk_count"]),
        ("带标题元数据", "不可用", docling_metrics["heading_metadata_chunk_count"]),
        ("识别为表格相关", "不可用", docling_metrics["table_aware_chunk_count"]),
        ("识别为图片相关", "不可用", docling_metrics["picture_aware_chunk_count"]),
    ]

    lines = [
        "# Fixed Window 与 Docling Hybrid 切分对比",
        "",
        "| 指标 | 当前固定窗口 | Docling Hybrid |",
        "|---|---:|---:|",
    ]
    lines.extend(f"| {name} | {fixed} | {docling} |" for name, fixed, docling in rows)
    lines.extend(
        [
            "",
            "## Docling Token 约束",
            "",
            f"- Token 数：min={docling_tokens['min']}，mean={docling_tokens['mean']}，"
            f"median={docling_tokens['median']}，P90={docling_tokens['p90']}，max={docling_tokens['max']}。",
            f"- 超出设定上限的输出块：{docling_metrics['over_token_limit_count']}。",
            "",
            "## 结构能力",
            "",
            f"- 跨页块索引：{docling_metrics['multi_page_chunk_indices']}。",
            f"- 表格相关块索引：{docling_metrics['table_chunk_indices']}。",
            f"- 图片相关块索引：{docling_metrics['picture_chunk_indices']}。",
            "",
            "## 解释边界",
            "",
            "- 字符长度与 token 长度不能直接等价；这里分别报告。",
            "- “疑似句中截断”仅根据末尾标点判断，表格、标题和列表会造成误判。",
            "- 本报告衡量切分结构，不等同于最终检索质量；是否采用仍需跑同一套 Recall/nDCG 评测。",
            "",
        ]
    )
    return "\n".join(lines)
