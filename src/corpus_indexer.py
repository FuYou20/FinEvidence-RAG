import json
from pathlib import Path
from .pdf_parser import parse_pdf
from .chunker import chunk_pages
from .embedder import embed_chunks
from .vector_store import build_index
REQUIRED_CORPUS_FIELDS = frozenset({"doc_id", "relative_path"})


def load_corpus_manifest(manifest_path):
    """读取并验证 JSONL 格式的语料清单。"""
    manifest_path = Path(manifest_path)
    records = []
    seen_doc_ids = set()

    # 文件不存在、无权限等异常已经具有明确语义，保留原始异常即可。
    with manifest_path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            stripped_line = raw_line.strip()
            if not stripped_line:
                continue

            # JSONDecodeError 属于底层格式错误，在 manifest 边界转换为
            # 带文件名和行号的 ValueError，便于调用方定位坏数据。
            try:
                record = json.loads(stripped_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"语料清单 {manifest_path} 第 {line_number} 行不是合法 JSON"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"语料清单 {manifest_path} 第 {line_number} 行必须是 JSON 对象"
                )

            missing_fields = REQUIRED_CORPUS_FIELDS - record.keys()
            if missing_fields:
                raise ValueError(
                    f"语料清单 {manifest_path} 第 {line_number} 行缺少必填字段: "
                    f"{sorted(missing_fields)}"
                )

            doc_id = record["doc_id"]
            relative_path = record["relative_path"]

            if not isinstance(doc_id, str) or not doc_id.strip():
                raise ValueError(
                    f"语料清单 {manifest_path} 第 {line_number} 行的 "
                    "doc_id 必须是非空字符串"
                )
            if not isinstance(relative_path, str) or not relative_path.strip():
                raise ValueError(
                    f"语料清单 {manifest_path} 第 {line_number} 行的 "
                    "relative_path 必须是非空字符串"
                )

            normalized_doc_id = doc_id.strip()
            if normalized_doc_id in seen_doc_ids:
                raise ValueError(
                    f"语料清单 {manifest_path} 第 {line_number} 行的 "
                    f"doc_id 重复: {normalized_doc_id}"
                )

            # 对标识和路径去除意外的首尾空白，但保留记录中的其他字段。
            normalized_record = dict(record)
            normalized_record["doc_id"] = normalized_doc_id
            normalized_record["relative_path"] = relative_path.strip()

            seen_doc_ids.add(normalized_doc_id)
            records.append(normalized_record)

    return records



def index_corpus(
    corpus_records,
    project_root,
    embed_batch,
    *,
    model_name,
    chunk_size=512,
    overlap=64,
    batch_size=20,
    include_tables=True,
):
    if not isinstance(corpus_records, list):
        raise TypeError("corpus_records 必须是列表")

    if not corpus_records:
        raise ValueError("corpus_records 不能为空")

    project_root = Path(project_root).resolve()
    if not project_root.is_dir():
        raise ValueError(f"project_root 不是有效目录: {project_root}")

    if not callable(embed_batch):
        raise TypeError("embed_batch 必须是可调用对象")

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name 必须是非空字符串")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size 必须是正整数")

    if not isinstance(overlap, int) or overlap < 0:
        raise ValueError("overlap 不能是负数")

    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")

    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size 必须是正整数")
    if not isinstance(include_tables, bool):
        raise TypeError("include_tables 必须是 bool")
    

    all_chunks = []

    for record in corpus_records:
        pdf_path = (project_root / record["relative_path"]).resolve()

        try:
            pdf_path.relative_to(project_root)
            
        except ValueError as exc:
            raise ValueError(f"PDF 路径越过项目目录: {record['relative_path']}") from exc
        chunk = parse_pdf(
            pdf_path=pdf_path,
            doc_id=record["doc_id"],
            include_tables=include_tables,
        )
        chunked_pages = chunk_pages(parsed_doc=chunk,chunk_size=chunk_size,overlap=overlap,)
        all_chunks.extend(chunked_pages)
    if not all_chunks:
        raise ValueError("语料库没有产生任何可索引文本块")
    embedded_chunks = embed_chunks(
            chunks=all_chunks,
            embed_batch=embed_batch,
            model_name=model_name,
            batch_size=batch_size,
        )
    index,metadata = build_index(embedded_chunks=embedded_chunks)
    return index ,metadata
