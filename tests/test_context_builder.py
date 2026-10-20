from copy import deepcopy

import pytest

from src.context_builder import build_context


def make_retrieval_results():
    # 故意按 rank=2、rank=1 的顺序构造，验证函数确实执行排序。
    return [
        {
            "doc_id": "report-001",
            "page_number": 20,
            "chunk_id": "report-001:p20:c0",
            "text": "传音通过本土化运营建立市场优势。",
            "rank": 2,
            "score": 0.76,
        },
        {
            "doc_id": "report-001",
            "page_number": 16,
            "chunk_id": "report-001:p16:c1",
            "text": "传音以高性价比产品开拓非洲市场。",
            "rank": 1,
            "score": 0.77,
        },
    ]


def test_build_context_returns_empty_string_for_empty_results():
    assert build_context([]) == ""


def test_build_context_sorts_results_and_adds_source_labels():
    context = build_context(make_retrieval_results())

    first_source_position = context.index("[S1]")
    second_source_position = context.index("[S2]")
    rank_one_text_position = context.index(
        "传音以高性价比产品开拓非洲市场。"
    )
    rank_two_text_position = context.index(
        "传音通过本土化运营建立市场优势。"
    )

    assert first_source_position < second_source_position
    assert first_source_position < rank_one_text_position
    assert rank_one_text_position < second_source_position
    assert second_source_position < rank_two_text_position


def test_build_context_includes_source_metadata():
    context = build_context(make_retrieval_results())

    assert "文档:report-001" in context
    assert "页码:16" in context
    assert "证据类型:正文" in context
    assert "块编号:report-001:p16:c1" in context
    assert "文本:传音以高性价比产品开拓非洲市场。" in context


def test_build_context_marks_table_evidence():
    results = make_retrieval_results()
    results[0]["content_type"] = "table"

    context = build_context(results)

    assert "证据类型:表格" in context


def test_build_context_separates_source_blocks():
    context = build_context(make_retrieval_results())

    assert "市场。\n\n[S2]" in context


def test_build_context_does_not_mutate_input():
    retrieval_results = make_retrieval_results()
    original_results = deepcopy(retrieval_results)

    build_context(retrieval_results)

    assert retrieval_results == original_results


@pytest.mark.parametrize(
    "missing_field",
    ["doc_id", "page_number", "chunk_id", "text", "rank"],
)
def test_build_context_raises_key_error_for_missing_required_field(
    missing_field,
):
    retrieval_results = make_retrieval_results()
    retrieval_results[0].pop(missing_field)

    with pytest.raises(KeyError, match=missing_field):
        build_context(retrieval_results)
