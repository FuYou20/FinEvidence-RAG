import json

import pytest

from src.evaluation.chunking_analysis import (
    analyze_docling_chunks,
    analyze_fixed_chunks,
    build_comparison_report,
    load_docling_chunks,
)


def test_load_docling_chunks_validates_jsonl(tmp_path):
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        json.dumps(
            {
                "chunk_index": 0,
                "text": "标题\n正文。",
                "num_tokens": 5,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_docling_chunks(path)[0]["chunk_index"] == 0


def test_analyze_fixed_chunks_reports_overlap_and_boundary():
    chunks = [
        {"text": "第一段没有结束"},
        {"text": "第二段结束。"},
    ]

    result = analyze_fixed_chunks(chunks, source_character_count=10)

    assert result["chunk_count"] == 2
    assert result["mid_sentence_end_count_heuristic"] == 1
    assert result["total_chunk_character_count"] > 10


def test_analyze_docling_chunks_reports_structure():
    chunks = [
        {
            "chunk_index": 0,
            "text": "章节标题\n正文。",
            "num_tokens": 20,
            "headings": ["章节标题"],
            "page_numbers": [1, 2],
            "doc_items": ["#/texts/1", "#/tables/0"],
        },
        {
            "chunk_index": 1,
            "text": "图注",
            "num_tokens": 600,
            "headings": [],
            "page_numbers": [2],
            "doc_items": ["#/pictures/0"],
        },
    ]

    result = analyze_docling_chunks(chunks, token_limit=512)

    assert result["multi_page_chunk_indices"] == [0]
    assert result["table_chunk_indices"] == [0]
    assert result["picture_chunk_indices"] == [1]
    assert result["over_token_limit_count"] == 1


def test_build_comparison_report_contains_core_metrics():
    fixed = analyze_fixed_chunks(
        [{"text": "固定窗口内容。"}],
        source_character_count=7,
    )
    docling = analyze_docling_chunks(
        [
            {
                "chunk_index": 0,
                "text": "结构化内容。",
                "num_tokens": 8,
                "headings": ["标题"],
                "page_numbers": [1],
                "doc_items": [],
            }
        ]
    )

    report = build_comparison_report(fixed, docling)

    assert "跨页 Chunk" in report
    assert "Docling Token 约束" in report


@pytest.mark.parametrize("token_limit", [0, -1])
def test_analyze_docling_chunks_rejects_invalid_limit(token_limit):
    with pytest.raises(ValueError):
        analyze_docling_chunks(
            [{"chunk_index": 0, "text": "x", "num_tokens": 1}],
            token_limit=token_limit,
        )
