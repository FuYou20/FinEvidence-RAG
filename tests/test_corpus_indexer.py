import json

import pytest

from src import corpus_indexer


def _write_jsonl(path, records):
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )


def _fake_parsed_document(pdf_path, doc_id, *, text=None):
    page_text = text if text is not None else f"{doc_id} 的测试证据"
    return {
        "doc_id": doc_id,
        "source_path": str(pdf_path),
        "page_count": 1,
        "text_page_count": 1 if page_text.strip() else 0,
        "status": "success" if page_text.strip() else "scanned_or_empty",
        "pages": [
            {
                "doc_id": doc_id,
                "page_number": 1,
                "text": page_text,
            }
        ],
    }


def _fake_embed_batch(texts):
    return [
        [float(index + 1), float(len(text)), 1.0]
        for index, text in enumerate(texts)
    ]


def test_load_corpus_manifest_returns_normalized_records(tmp_path):
    manifest_path = tmp_path / "corpus.jsonl"
    manifest_path.write_text(
        "\n"
        '{"doc_id": " doc_a ", "relative_path": " data/a.pdf ", "company": "A"}\n'
        '\n{"doc_id": "doc_b", "relative_path": "data/b.pdf", "company": "B"}\n',
        encoding="utf-8",
    )

    records = corpus_indexer.load_corpus_manifest(manifest_path)

    assert len(records) == 2
    assert records[0]["doc_id"] == "doc_a"
    assert records[0]["relative_path"] == "data/a.pdf"
    assert records[0]["company"] == "A"
    assert records[1]["doc_id"] == "doc_b"


def test_load_corpus_manifest_reports_invalid_json_line(tmp_path):
    manifest_path = tmp_path / "corpus.jsonl"
    manifest_path.write_text(
        '{"doc_id": "doc_a", "relative_path": "data/a.pdf"}\n'
        '{"doc_id": "doc_b",',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="第 2 行不是合法 JSON"):
        corpus_indexer.load_corpus_manifest(manifest_path)


def test_load_corpus_manifest_rejects_duplicate_doc_id(tmp_path):
    manifest_path = tmp_path / "corpus.jsonl"
    _write_jsonl(
        manifest_path,
        [
            {"doc_id": "doc_a", "relative_path": "data/a.pdf"},
            {"doc_id": "doc_a", "relative_path": "data/b.pdf"},
        ],
    )

    with pytest.raises(ValueError, match="doc_id 重复: doc_a"):
        corpus_indexer.load_corpus_manifest(manifest_path)


def test_index_corpus_builds_one_index_for_multiple_documents(
    tmp_path,
    monkeypatch,
):
    records = [
        {"doc_id": "doc_a", "relative_path": "data/a.pdf"},
        {"doc_id": "doc_b", "relative_path": "data/b.pdf"},
    ]

    def fake_parse_pdf(pdf_path, doc_id, *, include_tables):
        assert include_tables is True
        return _fake_parsed_document(pdf_path, doc_id)

    monkeypatch.setattr(corpus_indexer, "parse_pdf", fake_parse_pdf)

    index, metadata = corpus_indexer.index_corpus(
        corpus_records=records,
        project_root=tmp_path,
        embed_batch=_fake_embed_batch,
        model_name="fake-embedding-model",
        chunk_size=512,
        overlap=64,
        batch_size=2,
    )

    assert index.ntotal == len(metadata) == 2
    assert {item["doc_id"] for item in metadata} == {"doc_a", "doc_b"}
    assert {item["chunk_id"] for item in metadata} == {
        "doc_a:p1:c0",
        "doc_b:p1:c0",
    }
    assert {item["content_type"] for item in metadata} == {"text"}


def test_index_corpus_rejects_path_outside_project_root(
    tmp_path,
    monkeypatch,
):
    records = [
        {"doc_id": "outside", "relative_path": "../outside.pdf"},
    ]

    def parse_pdf_must_not_run(*args, **kwargs):
        raise AssertionError("路径校验失败时不应调用 parse_pdf")

    monkeypatch.setattr(
        corpus_indexer,
        "parse_pdf",
        parse_pdf_must_not_run,
    )

    with pytest.raises(ValueError, match="PDF 路径越过项目目录"):
        corpus_indexer.index_corpus(
            corpus_records=records,
            project_root=tmp_path,
            embed_batch=_fake_embed_batch,
            model_name="fake-embedding-model",
        )


def test_index_corpus_rejects_corpus_without_text_chunks(
    tmp_path,
    monkeypatch,
):
    records = [
        {"doc_id": "empty_doc", "relative_path": "data/empty.pdf"},
    ]

    def fake_parse_pdf(pdf_path, doc_id, *, include_tables):
        return _fake_parsed_document(pdf_path, doc_id, text="  \n\t")

    monkeypatch.setattr(corpus_indexer, "parse_pdf", fake_parse_pdf)

    with pytest.raises(ValueError, match="没有产生任何可索引文本块"):
        corpus_indexer.index_corpus(
            corpus_records=records,
            project_root=tmp_path,
            embed_batch=_fake_embed_batch,
            model_name="fake-embedding-model",
        )
