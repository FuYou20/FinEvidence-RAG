import json
from pathlib import Path

import pytest

from src.pdf_parser import parse_pdf


@pytest.fixture
def pdf_paths(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    pdf_dir = project_root / "data" / "pdf"

    return {
        "normal": (
            pdf_dir
            / "传音控股--奔跑的新兴市场，传音的长期主义--2022-04-22.pdf"
        ),
        "scanned": (
            pdf_dir
            / "中信证券--中信证券首次覆盖报告：优质龙头，更进一竿--2021-09-03.pdf"
        ),
        "missing": tmp_path / "not_exists.pdf",
    }


@pytest.fixture
def private_pdf_paths(pdf_paths):
    """Skip integration checks when proprietary report fixtures are absent."""
    required = (pdf_paths["normal"], pdf_paths["scanned"])
    if not all(path.is_file() for path in required):
        pytest.skip(
            "真实研报未随公开仓库分发；将测试 PDF 放入 data/pdf 后运行"
        )
    return pdf_paths


def test_read_normal_pdf(private_pdf_paths):
    result = parse_pdf(
        doc_id="test_001",
        pdf_path=private_pdf_paths["normal"],
    )

    assert result["page_count"] == 35
    assert result["text_page_count"] == 35
    assert result["status"] == "success"
    assert len(result["pages"]) == 35
    assert result["table_count"] >= 2
    page_27 = result["pages"][26]
    assert len(page_27["tables"]) == 1
    table = page_27["tables"][0]
    assert table["row_count"] == 11
    assert table["column_count"] == 6
    assert table["rows"][0][0] == "排名"
    assert "Google Play Games" in table["markdown"]
    assert "Google Play Games" not in page_27["text"]
    json.dumps(result, ensure_ascii=False)


def test_read_scanned_pdf(private_pdf_paths):
    result = parse_pdf(
        doc_id="test_002",
        pdf_path=private_pdf_paths["scanned"],
    )

    assert result["page_count"] == 26
    assert result["text_page_count"] == 0
    assert result["status"] == "scanned_or_empty"
    assert len(result["pages"]) == 26
    json.dumps(result, ensure_ascii=False)


def test_read_missing_pdf(pdf_paths):
    with pytest.raises(FileNotFoundError):
        parse_pdf(doc_id="test_003", pdf_path=pdf_paths["missing"])


def test_parse_pdf_can_disable_table_extraction(private_pdf_paths):
    result = parse_pdf(
        doc_id="test_no_tables",
        pdf_path=private_pdf_paths["normal"],
        include_tables=False,
    )

    assert result["table_count"] == 0
    assert all(not page["tables"] for page in result["pages"])


def test_parse_pdf_rejects_non_boolean_include_tables(pdf_paths):
    with pytest.raises(TypeError, match="include_tables"):
        parse_pdf(
            doc_id="test_invalid",
            pdf_path=pdf_paths["normal"],
            include_tables="yes",
        )
