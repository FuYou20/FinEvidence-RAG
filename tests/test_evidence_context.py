from copy import deepcopy

import pytest

from src.evidence_context import build_evidence_context


def make_results():
    return [
        {
            "doc_id": "report_1",
            "page_number": 3,
            "chunk_id": "report_1:p3:c0",
            "content_type": "text",
            "text": "公司收入保持增长。",
            "embedding": [0.1, 0.2],
            "score": 0.91,
        },
        {
            "doc_id": "report_1",
            "page_number": 5,
            "chunk_id": "report_1:p5:t0:c0",
            "content_type": "table",
            "table_id": "report_1:p5:t0",
            "text": "| 年份 | 收入 |\n|---|---|\n| 2022 | 100 |",
            "embedding": [0.3, 0.4],
            "score": 0.88,
        },
    ]


def test_builds_numbered_text_and_table_evidence():
    context, source_map = build_evidence_context(make_results())

    assert "[S1]" in context
    assert "证据类型：正文" in context
    assert "chunk_id：report_1:p3:c0" in context
    assert "[S2]" in context
    assert "证据类型：表格" in context
    assert "table_id：report_1:p5:t0" in context
    assert "| 2022 | 100 |" in context
    assert list(source_map) == ["S1", "S2"]
    assert source_map["S2"]["table_id"] == "report_1:p5:t0"


def test_source_map_is_independent_and_excludes_embedding():
    results = make_results()
    original = deepcopy(results)

    _, source_map = build_evidence_context(results)

    assert results == original
    assert "embedding" not in source_map["S1"]
    source_map["S1"]["text"] = "changed"
    assert results[0]["text"] == "公司收入保持增长。"


def test_missing_content_type_defaults_to_text():
    result = {
        "doc_id": "doc",
        "page_number": 1,
        "chunk_id": "doc:p1:c0",
        "text": "正文",
    }

    context, source_map = build_evidence_context([result])

    assert "证据类型：正文" in context
    assert source_map["S1"]["source_id"] == "S1"


def test_context_exposes_registry_id_and_web_provenance():
    item = {
        "doc_id": "web:example.com",
        "page_number": 1,
        "chunk_id": "web:1",
        "text": "搜索摘要",
        "evidence_id": "E-web-123",
        "source_kind": "web",
        "url": "https://example.com/report",
        "published_at": "2026-10-01",
        "content_origin": "search_summary",
    }
    context, _source_map = build_evidence_context([item])

    assert "evidence_id：E-web-123" in context
    assert "来源 URL：https://example.com/report" in context
    assert "发布日期：2026-10-01" in context
    assert "search_summary" in context
    assert "页码：1" not in context


@pytest.mark.parametrize("bad_input", [None, {}, "evidence"])
def test_rejects_non_list_input(bad_input):
    with pytest.raises(TypeError, match="必须是 list"):
        build_evidence_context(bad_input)


def test_rejects_empty_input():
    with pytest.raises(ValueError, match="不能为空"):
        build_evidence_context([])


def test_rejects_non_dict_evidence():
    with pytest.raises(TypeError, match="第 1 条证据必须是 dict"):
        build_evidence_context(["not-a-dict"])


def test_rejects_missing_required_field():
    result = {
        "doc_id": "doc",
        "page_number": 1,
        "text": "正文",
    }

    with pytest.raises(ValueError, match="chunk_id"):
        build_evidence_context([result])


@pytest.mark.parametrize("page_number", [0, -1, True, "1"])
def test_rejects_invalid_page_number(page_number):
    result = {
        "doc_id": "doc",
        "page_number": page_number,
        "chunk_id": "doc:p1:c0",
        "text": "正文",
    }

    with pytest.raises(ValueError, match="page_number"):
        build_evidence_context([result])


def test_table_evidence_requires_table_id():
    result = {
        "doc_id": "doc",
        "page_number": 1,
        "chunk_id": "doc:p1:t0:c0",
        "content_type": "table",
        "text": "| A | B |",
    }

    with pytest.raises(ValueError, match="table_id"):
        build_evidence_context([result])


def test_rejects_unsupported_content_type():
    result = {
        "doc_id": "doc",
        "page_number": 1,
        "chunk_id": "doc:p1:c0",
        "content_type": "image",
        "text": "图片描述",
    }

    with pytest.raises(ValueError, match="content_type"):
        build_evidence_context([result])
