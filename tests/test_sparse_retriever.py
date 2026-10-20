from copy import deepcopy

import pytest

from src.sparse_retriever import (
    build_bm25_index,
    sparse_retrieve,
    tokenize_text,
)


@pytest.fixture
def metadata():
    return [
        {
            "chunk_id": "doc:p1:c0",
            "doc_id": "doc",
            "page_number": 1,
            "text": "刀片电池可以降低电池成本，单车节约成本。",
        },
        {
            "chunk_id": "doc:p2:c0",
            "doc_id": "doc",
            "page_number": 2,
            "text": "公司手机业务在非洲市场快速发展。",
        },
        {
            "chunk_id": "doc:p3:c0",
            "doc_id": "doc",
            "page_number": 3,
            "text": "电解液业务贡献了新的利润增长。",
        },
    ]


def test_tokenize_text_normalizes_english_and_removes_punctuation():
    tokens = tokenize_text("  AI研报，成本分析！  ")

    assert "ai" in tokens
    assert "，" not in tokens
    assert "！" not in tokens
    assert all(token.strip() for token in tokens)


@pytest.mark.parametrize("text", [None, 1, []])
def test_tokenize_text_rejects_non_string(text):
    with pytest.raises(TypeError):
        tokenize_text(text)


@pytest.mark.parametrize("text", ["", "  \n\t"])
def test_tokenize_text_rejects_blank_text(text):
    with pytest.raises(ValueError):
        tokenize_text(text)


def test_build_and_sparse_retrieve_rank_keyword_match_first(metadata):
    index = build_bm25_index(metadata)

    results = sparse_retrieve(
        "刀片电池成本下降",
        index,
        metadata,
        top_k=2,
    )

    assert results[0]["chunk_id"] == "doc:p1:c0"
    assert results[0]["rank"] == 1
    assert results[0]["retrieval_source"] == "bm25"
    assert isinstance(results[0]["index_id"], int)
    assert isinstance(results[0]["score"], float)


def test_sparse_retrieve_does_not_mutate_metadata(metadata):
    original = deepcopy(metadata)
    index = build_bm25_index(metadata)

    sparse_retrieve("刀片电池成本", index, metadata)

    assert metadata == original


def test_sparse_retrieve_returns_empty_for_no_token_overlap(metadata):
    index = build_bm25_index(metadata)

    results = sparse_retrieve("量子引力黑洞", index, metadata)

    assert results == []


def test_custom_tokenizer_is_used_by_build_and_search():
    calls = []

    def tokenizer(text):
        calls.append(text)
        return text.lower().split()

    english_metadata = [
        {"chunk_id": "a", "text": "alpha beta"},
        {"chunk_id": "b", "text": "gamma delta"},
    ]
    index = build_bm25_index(english_metadata, tokenizer=tokenizer)
    sparse_retrieve(
        "alpha",
        index,
        english_metadata,
        tokenizer=tokenizer,
    )

    assert calls == ["alpha beta", "gamma delta", "alpha"]


@pytest.mark.parametrize(
    ("bad_metadata", "error_type"),
    [
        (None, TypeError),
        ({}, TypeError),
        ([], ValueError),
        (["not-a-dict"], TypeError),
        ([{}], ValueError),
        ([{"text": "  "}], ValueError),
    ],
)
def test_build_bm25_index_validates_metadata(bad_metadata, error_type):
    with pytest.raises(error_type):
        build_bm25_index(bad_metadata)


def test_build_bm25_index_accepts_tunable_parameters(metadata):
    index = build_bm25_index(
        metadata,
        k1=0.9,
        b=0.4,
        epsilon=0.1,
    )

    assert index.k1 == pytest.approx(0.9)
    assert index.b == pytest.approx(0.4)
    assert index.epsilon == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"k1": 0}, ValueError),
        ({"k1": True}, TypeError),
        ({"b": -0.1}, ValueError),
        ({"b": 1.1}, ValueError),
        ({"epsilon": -0.1}, ValueError),
        ({"epsilon": "0.25"}, TypeError),
    ],
)
def test_build_bm25_index_rejects_invalid_parameters(
    metadata,
    kwargs,
    error_type,
):
    with pytest.raises(error_type):
        build_bm25_index(metadata, **kwargs)


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_sparse_retrieve_rejects_invalid_top_k(metadata, top_k):
    index = build_bm25_index(metadata)

    with pytest.raises(ValueError):
        sparse_retrieve("成本", index, metadata, top_k=top_k)


def test_sparse_retrieve_rejects_score_count_mismatch(metadata):
    class BrokenIndex:
        def get_scores(self, _tokens):
            return [1.0]

    with pytest.raises(ValueError, match="分数数量"):
        sparse_retrieve("成本", BrokenIndex(), metadata)
