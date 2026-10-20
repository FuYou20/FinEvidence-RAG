import logging
from contextlib import nullcontext
from pathlib import Path

import pdfplumber
import pymupdf

from .table_parser import extract_page_tables


logger = logging.getLogger(__name__)

def _extract_text_lines(page):
    lines = []
    # Keep PyMuPDF's native content-stream order.  This matches the original
    # page.get_text() pipeline and avoids changing existing chunk boundaries
    # merely because table extraction also needs line bounding boxes.
    page_dict = page.get_text("dict", sort=False)
    for block in page_dict["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(
                span.get("text", "")
                for span in line.get("spans", [])
            ).strip()
            if not text:
                continue
            lines.append(
                {
                    "text": text,
                    "bbox": [float(value) for value in line["bbox"]],
                }
            )
    return lines


def _intersection_ratio(subject_bbox, container_bbox):
    x0 = max(subject_bbox[0], container_bbox[0])
    y0 = max(subject_bbox[1], container_bbox[1])
    x1 = min(subject_bbox[2], container_bbox[2])
    y1 = min(subject_bbox[3], container_bbox[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    subject_area = max(0.0, subject_bbox[2] - subject_bbox[0]) * max(
        0.0,
        subject_bbox[3] - subject_bbox[1],
    )
    return intersection / subject_area if subject_area > 0 else 0.0


def _remove_table_lines(text_lines, tables):
    table_bboxes = [table["bbox"] for table in tables]
    return [
        line
        for line in text_lines
        if not any(
            _intersection_ratio(line["bbox"], table_bbox) >= 0.5
            for table_bbox in table_bboxes
        )
    ]


def parse_pdf(pdf_path, doc_id, *, include_tables=True):
    """Parse text pages and optionally extract ruled tables as elements."""
    if not isinstance(include_tables, bool):
        raise TypeError("include_tables 必须是 bool")

    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF 文件不存在或不是文件: {pdf_path}")

    plumber_context = (
        pdfplumber.open(pdf_path)
        if include_tables
        else nullcontext(None)
    )
    with pymupdf.open(pdf_path) as doc, plumber_context as plumber_doc:
        page_count = doc.page_count
        text_page_count = 0
        content_page_count = 0
        table_count = 0
        table_error_count = 0
        pages = []

        for page_index, page in enumerate(doc):
            page_number = page_index + 1
            raw_text = page.get_text().strip()
            text_lines = _extract_text_lines(page) if include_tables else []
            tables = []
            table_extraction_error = None

            if include_tables:
                try:
                    tables = extract_page_tables(
                        plumber_doc.pages[page_index],
                        doc_id=doc_id,
                        page_number=page_number,
                        text_lines=text_lines,
                        target_width=page.rect.width,
                        target_height=page.rect.height,
                    )
                except Exception as exc:
                    table_error_count += 1
                    table_extraction_error = f"{type(exc).__name__}: {exc}"
                    logger.warning(
                        "表格提取失败，回退到纯文本: doc_id=%s page=%s error=%s",
                        doc_id,
                        page_number,
                        type(exc).__name__,
                    )

            if tables:
                body_lines = _remove_table_lines(text_lines, tables)
                text = "\n".join(
                    line["text"] for line in body_lines
                ).strip()
            else:
                # Preserve the pre-table-pipeline representation exactly on
                # ordinary pages so their chunk boundaries and embeddings do
                # not drift.  Extraction failures also take this safe path.
                text = raw_text
            if text:
                text_page_count += 1
            if text or tables:
                content_page_count += 1
            table_count += len(tables)

            pages.append(
                {
                    "doc_id": doc_id,
                    "page_number": page_number,
                    "text": text,
                    "tables": tables,
                    "table_extraction_error": table_extraction_error,
                }
            )

    return {
        "doc_id": doc_id,
        "source_path": str(pdf_path),
        "page_count": page_count,
        "text_page_count": text_page_count,
        "content_page_count": content_page_count,
        "table_count": table_count,
        "table_error_count": table_error_count,
        "status": "success" if content_page_count > 0 else "scanned_or_empty",
        "pages": pages,
    }
