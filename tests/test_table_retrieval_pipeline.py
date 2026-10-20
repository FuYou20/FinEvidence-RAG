from src.chunker import chunk_pages
from src.context_builder import build_context
from src.embedder import embed_chunks
from src.retriever import retrieve
from src.sparse_retriever import build_bm25_index, sparse_retrieve
from src.vector_store import build_index


def make_parsed_document_with_table():
    return {
        "doc_id": "report-1",
        "source_path": "report.pdf",
        "pages": [
            {
                "doc_id": "report-1",
                "page_number": 1,
                "text": "这是一段与排行榜无关的普通正文。",
                "tables": [
                    {
                        "table_id": "report-1:p1:t0",
                        "table_index": 0,
                        "bbox": [10.0, 20.0, 500.0, 300.0],
                        "caption": "图表1：非洲音乐APP排行榜",
                        "rows": [
                            ["APP", "周人均使用时长"],
                            ["Boomplay", "40.78分钟"],
                            ["SoundCloud", "22.95分钟"],
                        ],
                        "row_count": 3,
                        "column_count": 2,
                        "extraction_method": "pdfplumber_lines_v2",
                    }
                ],
            },
            {
                "doc_id": "report-1",
                "page_number": 2,
                "text": "公司持续建设销售渠道并提升品牌影响力。",
                "tables": [],
            },
            {
                "doc_id": "report-1",
                "page_number": 3,
                "text": "新兴市场智能手机需求保持增长。",
                "tables": [],
            },
        ],
    }


def test_table_chunk_flows_through_bm25_and_context_builder():
    chunks = chunk_pages(
        make_parsed_document_with_table(),
        chunk_size=512,
        overlap=64,
    )
    bm25_index = build_bm25_index(chunks)

    results = sparse_retrieve(
        "Boomplay 周人均使用时长",
        bm25_index,
        chunks,
        top_k=1,
    )
    context = build_context(results)

    assert results[0]["content_type"] == "table"
    assert results[0]["table_id"] == "report-1:p1:t0"
    assert "40.78分钟" in results[0]["text"]
    assert "证据类型:表格" in context


def test_table_chunk_flows_through_dense_index_and_retrieval():
    chunks = chunk_pages(
        make_parsed_document_with_table(),
        chunk_size=512,
        overlap=64,
    )

    def fake_embed_batch(texts):
        return [
            [0.0, 1.0] if "Boomplay" in text else [1.0, 0.0]
            for text in texts
        ]

    embedded = embed_chunks(
        chunks,
        fake_embed_batch,
        model_name="fake-model",
    )
    index, metadata = build_index(embedded)
    results = retrieve(
        query="Boomplay",
        index=index,
        metadata=metadata,
        embed_batch=fake_embed_batch,
        top_k=1,
    )

    assert results[0]["content_type"] == "table"
    assert results[0]["table_id"] == "report-1:p1:t0"
