"""Deterministic extraction and normalization of text-layer PDF tables."""

import re


TABLE_EXTRACTION_VERSION = "pdfplumber_lines_v4"
TABLE_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
}
CAPTION_PATTERN = re.compile(r"(?:图表|表)\s*\d+", re.IGNORECASE)


def clean_cell(value):
    if value is None:
        return ""
    return " ".join(str(value).replace("\n", " ").split())


def normalize_rows(rows):
    if not rows:
        return []
    normalized = [[clean_cell(cell) for cell in row] for row in rows]
    column_count = max((len(row) for row in normalized), default=0)
    return [row + [""] * (column_count - len(row)) for row in normalized]


def table_quality(rows):
    row_count = len(rows)
    column_count = max((len(row) for row in rows), default=0)
    total_cell_count = row_count * column_count
    nonempty_cell_count = sum(
        bool(cell) for row in rows for cell in row
    )
    nonempty_ratio = (
        nonempty_cell_count / total_cell_count if total_cell_count else 0.0
    )
    informative_column_count = sum(
        sum(bool(row[column_index]) for row in rows) >= 2
        for column_index in range(column_count)
    )
    nonempty_row_count = sum(any(row) for row in rows)
    return {
        "row_count": row_count,
        "column_count": column_count,
        "nonempty_cell_count": nonempty_cell_count,
        "nonempty_ratio": nonempty_ratio,
        "informative_column_count": informative_column_count,
        "nonempty_row_count": nonempty_row_count,
    }


def is_meaningful_table(rows):
    quality = table_quality(rows)
    return (
        quality["row_count"] >= 2
        and quality["column_count"] >= 2
        and quality["nonempty_cell_count"] >= 4
        and quality["nonempty_ratio"] >= 0.3
        and quality["informative_column_count"] >= 2
        and quality["nonempty_row_count"] >= 2
    )


def table_to_markdown(rows):
    if not rows:
        return ""

    def render_row(row):
        escaped = [
            cell.replace("\\", "\\\\").replace("|", "\\|")
            for cell in row
        ]
        return "| " + " | ".join(escaped) + " |"

    return "\n".join(
        [
            render_row(rows[0]),
            render_row(["---"] * len(rows[0])),
            *(render_row(row) for row in rows[1:]),
        ]
    )


def _horizontal_overlap_ratio(first_bbox, second_bbox):
    overlap = max(
        0.0,
        min(first_bbox[2], second_bbox[2])
        - max(first_bbox[0], second_bbox[0]),
    )
    smaller_width = min(
        first_bbox[2] - first_bbox[0],
        second_bbox[2] - second_bbox[0],
    )
    return overlap / smaller_width if smaller_width > 0 else 0.0


def find_table_caption(text_lines, table_bbox, *, max_distance=60.0):
    candidates = []
    for line in text_lines:
        text = line["text"].strip()
        bbox = line["bbox"]
        distance = table_bbox[1] - bbox[3]
        if not 0 <= distance <= max_distance:
            continue
        if not CAPTION_PATTERN.search(text):
            continue
        if _horizontal_overlap_ratio(bbox, table_bbox) < 0.2:
            continue
        candidates.append((distance, -bbox[1], text))

    return min(candidates)[2] if candidates else ""


def extract_page_tables(
    plumber_page,
    *,
    doc_id,
    page_number,
    text_lines,
    target_width,
    target_height,
):
    """Extract useful ruled tables and map bboxes into PyMuPDF coordinates."""
    scale_x = target_width / float(plumber_page.width)
    scale_y = target_height / float(plumber_page.height)
    extracted = []

    for detected in plumber_page.find_tables(table_settings=TABLE_SETTINGS):
        rows = normalize_rows(detected.extract())
        if not is_meaningful_table(rows):
            continue

        x0, top, x1, bottom = detected.bbox
        bbox = [
            float(x0) * scale_x,
            float(top) * scale_y,
            float(x1) * scale_x,
            float(bottom) * scale_y,
        ]
        table_index = len(extracted)
        quality = table_quality(rows)
        caption = find_table_caption(text_lines, bbox)
        markdown = table_to_markdown(rows)
        extracted.append(
            {
                "table_id": f"{doc_id}:p{page_number}:t{table_index}",
                "table_index": table_index,
                "bbox": bbox,
                "caption": caption,
                "rows": rows,
                "markdown": markdown,
                "extraction_method": TABLE_EXTRACTION_VERSION,
                **quality,
            }
        )

    return extracted
