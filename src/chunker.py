from .table_parser import table_to_markdown


def chunk_text(text, chunk_size=512, overlap=64):
    """按照固定字符窗口切分文本，并保留原文字符区间。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0:
        raise ValueError("overlap 不能小于 0")
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")
    if not text.strip():
        return []

    chunks = []
    start = 0
    text_len = len(text)
    step = chunk_size - overlap

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(
                {
                    "chunk_index": len(chunks),
                    "text": chunk,
                    "char_start": start,
                    "char_end": end,
                }
            )

        if end == text_len:
            break
        start += step

    return chunks


def _render_table_text(caption, header, data_rows):
    markdown = table_to_markdown([header, *data_rows])
    if caption:
        return f"表格标题：{caption}\n\n{markdown}"
    return markdown


def chunk_table(table, *, chunk_size=512):
    """按数据行切分表格，并在每个子块中重复表头。"""
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise TypeError("chunk_size 必须是整数")
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")

    rows = table.get("rows", [])
    if not rows:
        return []
    header = rows[0]
    caption = table.get("caption", "")
    indexed_data_rows = list(enumerate(rows[1:], start=1))

    if not indexed_data_rows:
        return [
            {
                "table_chunk_index": 0,
                "table_row_start": 0,
                "table_row_end": 0,
                "text": _render_table_text(caption, header, []),
            }
        ]

    chunks = []
    current_rows = []
    current_indices = []

    def flush():
        if not current_rows:
            return
        chunks.append(
            {
                "table_chunk_index": len(chunks),
                "table_row_start": current_indices[0],
                "table_row_end": current_indices[-1],
                "text": _render_table_text(caption, header, current_rows),
            }
        )

    for row_index, row in indexed_data_rows:
        candidate_rows = [*current_rows, row]
        candidate_text = _render_table_text(caption, header, candidate_rows)
        if current_rows and len(candidate_text) > chunk_size:
            flush()
            current_rows = [row]
            current_indices = [row_index]
        else:
            current_rows.append(row)
            current_indices.append(row_index)

    flush()
    return chunks


def chunk_pages(parsed_doc, chunk_size=512, overlap=64):
    """将正文和表格元素转换成统一的可检索 chunk。"""
    doc_id = parsed_doc["doc_id"]
    source_path = parsed_doc["source_path"]
    chunk_list = []

    for page in parsed_doc["pages"]:
        page_number = page["page_number"]
        text_chunks = chunk_text(
            text=page["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        )
        for chunk in text_chunks:
            chunk_index = chunk["chunk_index"]
            chunk_list.append(
                {
                    "chunk_id": f"{doc_id}:p{page_number}:c{chunk_index}",
                    "doc_id": doc_id,
                    "source_path": source_path,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "document_chunk_index": len(chunk_list),
                    "content_type": "text",
                    "char_start": chunk["char_start"],
                    "char_end": chunk["char_end"],
                    "text": chunk["text"],
                    "chunking_version": f"char_{chunk_size}_overlap_{overlap}",
                }
            )

        for table in page.get("tables", []):
            for part in chunk_table(table, chunk_size=chunk_size):
                table_index = table["table_index"]
                part_index = part["table_chunk_index"]
                chunk_list.append(
                    {
                        "chunk_id": (
                            f"{doc_id}:p{page_number}:"
                            f"t{table_index}:c{part_index}"
                        ),
                        "doc_id": doc_id,
                        "source_path": source_path,
                        "page_number": page_number,
                        "chunk_index": part_index,
                        "document_chunk_index": len(chunk_list),
                        "content_type": "table",
                        "char_start": None,
                        "char_end": None,
                        "text": part["text"],
                        "chunking_version": f"table_rows_v1_char_{chunk_size}",
                        "table_id": table["table_id"],
                        "table_index": table_index,
                        "table_chunk_index": part_index,
                        "table_bbox": table["bbox"],
                        "table_caption": table["caption"],
                        "table_row_start": part["table_row_start"],
                        "table_row_end": part["table_row_end"],
                        "table_total_rows": table["row_count"],
                        "table_total_columns": table["column_count"],
                        "extraction_method": table["extraction_method"],
                    }
                )

    return chunk_list
