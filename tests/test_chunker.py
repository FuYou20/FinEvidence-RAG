import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.chunker import chunk_pages, chunk_table, chunk_text
from src.pdf_parser import parse_pdf


def make_parsed_doc():
    return {
        "doc_id": "test_doc",
        "source_path": "test.pdf",
        "pages": [
            {"doc_id": "test_doc", "page_number": 1, "text": "abcdefghij"},
            {"doc_id": "test_doc", "page_number": 2, "text": "   "},
            {"doc_id": "test_doc", "page_number": 3, "text": "xyz"},
        ],
    }


def test_short_text_creates_one_chunk():
    text = "abc"

    chunks = chunk_text(text, chunk_size=6, overlap=2)

    assert chunks == [
        {
            "chunk_index": 0,
            "text": "abc",
            "char_start": 0,
            "char_end": 3,
        }
    ]


def test_long_text_creates_overlapping_chunks():
    text = "abcdefghij"

    chunks = chunk_text(text, chunk_size=6, overlap=2)

    assert len(chunks) == 2
    assert chunks[0]["text"] == "abcdef"
    assert chunks[1]["text"] == "efghij"
    assert chunks[0]["text"][-2:] == chunks[1]["text"][:2]
    assert chunks[0]["char_start"] == 0
    assert chunks[0]["char_end"] == 6
    assert chunks[1]["char_start"] == 4
    assert chunks[1]["char_end"] == 10


@pytest.mark.parametrize("text", ["", "   \n\t"])
def test_blank_text_returns_no_chunks(text):
    assert chunk_text(text, chunk_size=6, overlap=2) == []


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (0, 0),
        (-1, 0),
        (6, -1),
        (6, 6),
        (6, 7),
    ],
)
def test_invalid_window_parameters_raise_value_error(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_text("abcdefghij", chunk_size=chunk_size, overlap=overlap)


def test_every_chunk_matches_its_source_offsets():
    text = "0123456789ABC"

    chunks = chunk_text(text, chunk_size=6, overlap=2)

    assert chunks[0]["char_start"] == 0
    assert chunks[-1]["char_end"] == len(text)
    for chunk in chunks:
        start = chunk["char_start"]
        end = chunk["char_end"]
        assert chunk["text"] == text[start:end]
        assert len(chunk["text"]) <= 6


def test_chunk_pages_attaches_metadata_and_skips_blank_pages():
    chunks = chunk_pages(make_parsed_doc(), chunk_size=6, overlap=2)

    assert len(chunks) == 3
    assert [chunk["chunk_id"] for chunk in chunks] == [
        "test_doc:p1:c0",
        "test_doc:p1:c1",
        "test_doc:p3:c0",
    ]
    assert [chunk["page_number"] for chunk in chunks] == [1, 1, 3]
    assert [chunk["chunk_index"] for chunk in chunks] == [0, 1, 0]
    assert [chunk["document_chunk_index"] for chunk in chunks] == [0, 1, 2]
    assert all(chunk["doc_id"] == "test_doc" for chunk in chunks)
    assert all(chunk["source_path"] == "test.pdf" for chunk in chunks)
    assert all(chunk["content_type"] == "text" for chunk in chunks)
    assert all(
        chunk["chunking_version"] == "char_6_overlap_2"
        for chunk in chunks
    )


def test_chunk_pages_offsets_match_their_source_pages():
    parsed_doc = make_parsed_doc()
    page_text = {
        page["page_number"]: page["text"]
        for page in parsed_doc["pages"]
    }

    chunks = chunk_pages(parsed_doc, chunk_size=6, overlap=2)

    for chunk in chunks:
        if chunk["content_type"] != "text":
            continue
        source_text = page_text[chunk["page_number"]]
        assert chunk["text"] == source_text[
            chunk["char_start"]:chunk["char_end"]
        ]


def test_chunk_pages_does_not_mutate_input_and_is_json_serializable():
    parsed_doc = make_parsed_doc()
    original = deepcopy(parsed_doc)

    chunks = chunk_pages(parsed_doc, chunk_size=6, overlap=2)

    assert parsed_doc == original
    json.dumps(chunks, ensure_ascii=False)


def test_real_pdf_parse_and_chunk_pipeline():
    project_root = Path(__file__).resolve().parents[1]
    pdf_path = (
        project_root
        / "data"
        / "pdf"
        / "传音控股--奔跑的新兴市场，传音的长期主义--2022-04-22.pdf"
    )
    if not pdf_path.is_file():
        pytest.skip(
            "真实研报未随公开仓库分发；将测试 PDF 放入 data/pdf 后运行"
        )
    parsed_doc = parse_pdf(pdf_path=pdf_path, doc_id="integration_report")

    chunks = chunk_pages(parsed_doc, chunk_size=512, overlap=64)

    assert chunks
    assert all(chunk["doc_id"] == "integration_report" for chunk in chunks)
    assert all(chunk["source_path"] == str(pdf_path) for chunk in chunks)
    pages = {
        page["page_number"]: page["text"]
        for page in parsed_doc["pages"]
    }
    for chunk in chunks:
        if chunk["content_type"] != "text":
            continue
        page_text = pages[chunk["page_number"]]
        assert chunk["text"] == page_text[
            chunk["char_start"]:chunk["char_end"]
        ]


def make_table():
    return {
        "table_id": "test_doc:p1:t0",
        "table_index": 0,
        "bbox": [10.0, 20.0, 300.0, 200.0],
        "caption": "图表1：测试表格",
        "rows": [
            ["年份", "营收"],
            ["2021", "100"],
            ["2022", "120"],
            ["2023", "150"],
        ],
        "row_count": 4,
        "column_count": 2,
        "extraction_method": "pdfplumber_lines_v1",
    }


def test_chunk_table_splits_by_rows_and_repeats_header():
    chunks = chunk_table(make_table(), chunk_size=75)

    assert len(chunks) >= 2
    assert all("| 年份 | 营收 |" in chunk["text"] for chunk in chunks)
    assert all("图表1：测试表格" in chunk["text"] for chunk in chunks)
    assert chunks[0]["table_row_start"] == 1
    assert chunks[-1]["table_row_end"] == 3


def test_chunk_pages_adds_table_chunks_with_traceable_metadata():
    parsed_doc = make_parsed_doc()
    parsed_doc["pages"][0]["tables"] = [make_table()]

    chunks = chunk_pages(parsed_doc, chunk_size=75, overlap=10)
    table_chunks = [
        chunk for chunk in chunks if chunk["content_type"] == "table"
    ]

    assert table_chunks
    assert all(chunk["table_id"] == "test_doc:p1:t0" for chunk in table_chunks)
    assert all(chunk["char_start"] is None for chunk in table_chunks)
    assert all(":t0:c" in chunk["chunk_id"] for chunk in table_chunks)
    assert [chunk["document_chunk_index"] for chunk in chunks] == list(
        range(len(chunks))
    )
