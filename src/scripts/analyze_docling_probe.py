"""Compare one Docling chunk export with the current fixed-window chunker."""

import argparse
import json
from pathlib import Path

from src.chunker import chunk_pages
from src.evaluation.chunking_analysis import (
    analyze_docling_chunks,
    analyze_fixed_chunks,
    build_comparison_report,
    load_docling_chunks,
)
from src.pdf_parser import parse_pdf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--docling-jsonl", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--token-limit", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()

    parsed_doc = parse_pdf(args.pdf, args.doc_id)
    fixed_chunks = chunk_pages(
        parsed_doc,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    source_character_count = sum(
        len(page["text"]) for page in parsed_doc["pages"]
    )
    docling_chunks = load_docling_chunks(args.docling_jsonl)

    comparison = {
        "pdf": str(args.pdf),
        "fixed": analyze_fixed_chunks(
            fixed_chunks,
            source_character_count=source_character_count,
        ),
        "docling": analyze_docling_chunks(
            docling_chunks,
            token_limit=args.token_limit,
        ),
    }

    output_json = args.output_prefix.with_suffix(".json")
    output_markdown = args.output_prefix.with_suffix(".md")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    output_markdown.write_text(
        build_comparison_report(comparison["fixed"], comparison["docling"]),
        encoding="utf-8",
    )

    print(output_json)
    print(output_markdown)


if __name__ == "__main__":
    main()
