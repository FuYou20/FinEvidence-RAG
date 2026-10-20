from copy import deepcopy

import pytest

from errors import EmptyDocumentError, QueryEmbeddingError, RAGError
import src.retriever as retriever


def make_parsed_doc():
    return {
        "doc_id": "report-001",
        "source_path": "/fake/report.pdf",
        "page_count": 1,
        "text_page_count": 1,
        "status": "success",
        "pages": [
            {
                "doc_id": "report-001",
                "page_number": 1,
                "text": "第一段文本。第二段文本。",
            }
        ],
    }


def make_chunks():
    return [
        {
            "chunk_id": "report-001:p1:c0",
            "doc_id": "report-001",
            "source_path": "/fake/report.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "char_start": 0,
            "char_end": 6,
            "text": "第一段文本",
            "chunking_version": "char_6_overlap_0",
        },
        {
            "chunk_id": "report-001:p1:c1",
            "doc_id": "report-001",
            "source_path": "/fake/report.pdf",
            "page_number": 1,
            "chunk_index": 1,
            "char_start": 6,
            "char_end": 12,
            "text": "第二段文本",
            "chunking_version": "char_6_overlap_0",
        },
    ]


def test_empty_document_error_is_a_rag_error():
    assert issubclass(EmptyDocumentError, RAGError)
    assert issubclass(QueryEmbeddingError, RAGError)


def test_index_document_composes_pipeline_without_real_api(monkeypatch):
    parsed_doc = make_parsed_doc()
    chunks = make_chunks()
    original_chunks = deepcopy(chunks)
    embed_calls = []

    def fake_parse_pdf(*, pdf_path, doc_id, include_tables):
        assert pdf_path == "/fake/report.pdf"
        assert doc_id == "report-001"
        assert include_tables is True
        return parsed_doc

    def fake_chunk_pages(*, parsed_doc, chunk_size, overlap):
        assert parsed_doc is not None
        assert chunk_size == 6
        assert overlap == 0
        return chunks

    def fake_embed_batch(texts):
        embed_calls.append(list(texts))
        return [
            [float(len(text)), 1.0]
            for text in texts
        ]

    monkeypatch.setattr(retriever, "parse_pdf", fake_parse_pdf)
    monkeypatch.setattr(retriever, "chunk_pages", fake_chunk_pages)

    index, metadata = retriever.index_document(
        pdf_path="/fake/report.pdf",
        doc_id="report-001",
        embed_batch=fake_embed_batch,
        model_name="fake-model",
        chunk_size=6,
        overlap=0,
        batch_size=20,
    )

    assert embed_calls == [["第一段文本", "第二段文本"]]
    assert index.ntotal == 2
    assert len(metadata) == 2
    assert [item["chunk_id"] for item in metadata] == [
        "report-001:p1:c0",
        "report-001:p1:c1",
    ]
    assert chunks == original_chunks


def test_index_document_rejects_document_without_chunks(monkeypatch):
    monkeypatch.setattr(
        retriever,
        "parse_pdf",
        lambda **_kwargs: make_parsed_doc(),
    )
    monkeypatch.setattr(
        retriever,
        "chunk_pages",
        lambda **_kwargs: [],
    )

    def must_not_embed(_texts):
        raise AssertionError("空文档不应调用 Embedding Provider")

    with pytest.raises(EmptyDocumentError, match="report-empty"):
        retriever.index_document(
            pdf_path="/fake/empty.pdf",
            doc_id="report-empty",
            embed_batch=must_not_embed,
            model_name="fake-model",
        )


def test_index_document_preserves_file_not_found_error(monkeypatch):
    def raise_file_not_found(**_kwargs):
        raise FileNotFoundError("missing.pdf")

    monkeypatch.setattr(retriever, "parse_pdf", raise_file_not_found)

    with pytest.raises(FileNotFoundError, match="missing.pdf"):
        retriever.index_document(
            pdf_path="/fake/missing.pdf",
            doc_id="report-missing",
            embed_batch=lambda texts: texts,
            model_name="fake-model",
        )


def test_retrieve_embeds_once_and_passes_vector_to_search(monkeypatch):
    embed_calls = []
    search_calls = []
    expected_results = [{"chunk_id": "report-001:p1:c0", "rank": 1}]
    fake_index = object()
    fake_metadata = [{"chunk_id": "report-001:p1:c0"}]

    def fake_embed_batch(texts):
        embed_calls.append(list(texts))
        return [[0.25, 0.75]]

    def fake_search_index(*, index, metadata, query_vector, top_k):
        search_calls.append(
            {
                "index": index,
                "metadata": metadata,
                "query_vector": query_vector,
                "top_k": top_k,
            }
        )
        return expected_results

    monkeypatch.setattr(retriever, "search_index", fake_search_index)

    result = retriever.retrieve(
        query="  测试问题 \n",
        index=fake_index,
        metadata=fake_metadata,
        embed_batch=fake_embed_batch,
        top_k=3,
    )

    assert result is expected_results
    assert embed_calls == [["测试问题"]]
    assert search_calls == [
        {
            "index": fake_index,
            "metadata": fake_metadata,
            "query_vector": [0.25, 0.75],
            "top_k": 3,
        }
    ]


def test_retrieve_can_expand_neighbor_chunks(monkeypatch):
    metadata = [
        {
            "doc_id": "doc-a",
            "chunk_id": f"doc-a:p1:c{index}",
            "document_chunk_index": index,
            "page_number": 1,
            "text": f"chunk-{index}",
        }
        for index in range(3)
    ]

    def fake_search_index(**_kwargs):
        return [
            {
                **metadata[1],
                "rank": 1,
                "score": 1.0,
            }
        ]

    monkeypatch.setattr(retriever, "search_index", fake_search_index)

    results = retriever.retrieve(
        query="测试问题",
        index=object(),
        metadata=metadata,
        embed_batch=lambda _texts: [[1.0, 0.0]],
        top_k=3,
        neighbor_window=1,
        neighbor_seed_top_n=1,
        neighbor_decay=0.8,
    )

    assert [item["chunk_id"] for item in results] == [
        "doc-a:p1:c1",
        "doc-a:p1:c0",
        "doc-a:p1:c2",
    ]
    assert results[0]["is_direct_hit"] is True
    assert all(item["is_direct_hit"] is False for item in results[1:])


@pytest.mark.parametrize("neighbor_window", [-1, 1.5, True])
def test_retrieve_rejects_invalid_neighbor_window(
    monkeypatch,
    neighbor_window,
):
    monkeypatch.setattr(retriever, "search_index", lambda **_kwargs: [])

    expected_error = TypeError if neighbor_window in (1.5, True) else ValueError
    with pytest.raises(expected_error, match="neighbor_window"):
        retriever.retrieve(
            query="测试问题",
            index=object(),
            metadata=[],
            embed_batch=lambda _texts: [[1.0]],
            neighbor_window=neighbor_window,
        )


@pytest.mark.parametrize("query", [None, 123, ["问题"]])
def test_retrieve_rejects_non_string_query(query):
    with pytest.raises(TypeError, match="query 必须是字符串"):
        retriever.retrieve(
            query=query,
            index=object(),
            metadata=[],
            embed_batch=lambda _texts: [[1.0]],
        )


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_retrieve_rejects_blank_query(query):
    with pytest.raises(ValueError, match="query 不能为空"):
        retriever.retrieve(
            query=query,
            index=object(),
            metadata=[],
            embed_batch=lambda _texts: [[1.0]],
        )


@pytest.mark.parametrize(
    ("provider_result", "expected_count"),
    [
        ([], 0),
        ([[1.0, 0.0], [0.0, 1.0]], 2),
    ],
)
def test_retrieve_rejects_wrong_number_of_query_vectors(
    provider_result,
    expected_count,
):
    with pytest.raises(
        QueryEmbeddingError,
        match=f"实际得到 {expected_count} 个",
    ):
        retriever.retrieve(
            query="测试问题",
            index=object(),
            metadata=[],
            embed_batch=lambda _texts: provider_result,
        )


def test_retrieve_rejects_non_collection_provider_result():
    with pytest.raises(
        QueryEmbeddingError,
        match="必须返回向量列表",
    ):
        retriever.retrieve(
            query="测试问题",
            index=object(),
            metadata=[],
            embed_batch=lambda _texts: None,
        )


def test_retrieve_preserves_provider_exception():
    def unavailable_provider(_texts):
        raise TimeoutError("embedding timeout")

    with pytest.raises(TimeoutError, match="embedding timeout"):
        retriever.retrieve(
            query="测试问题",
            index=object(),
            metadata=[],
            embed_batch=unavailable_provider,
        )
