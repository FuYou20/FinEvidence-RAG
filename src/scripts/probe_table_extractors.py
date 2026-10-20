"""Compare pdfplumber and Camelot on representative research-report pages."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import camelot
import pdfplumber


DEFAULT_PAGES = (16, 27, 28)


def _clean_cell(value):
    if value is None:
        return ""
    return " ".join(str(value).replace("\n", " ").split())


def _table_summary(data, *, extra=None):
    rows = [[_clean_cell(cell) for cell in row] for row in data]
    column_count = max((len(row) for row in rows), default=0)
    nonempty_count = sum(bool(cell) for row in rows for cell in row)
    result = {
        "row_count": len(rows),
        "column_count": column_count,
        "nonempty_cell_count": nonempty_count,
        "preview": rows[:3],
    }
    if extra:
        result.update(extra)
    return result


def probe_pdfplumber(pdf_path, pages, *, strategy):
    settings = None
    if strategy == "text":
        settings = {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "min_words_vertical": 3,
            "min_words_horizontal": 1,
        }

    started_at = perf_counter()
    page_results = []
    with pdfplumber.open(pdf_path) as document:
        for page_number in pages:
            page = document.pages[page_number - 1]
            tables = page.find_tables(table_settings=settings)
            page_results.append(
                {
                    "page_number": page_number,
                    "table_count": len(tables),
                    "tables": [
                        _table_summary(
                            table.extract(),
                            extra={"bbox": list(table.bbox)},
                        )
                        for table in tables
                    ],
                }
            )
    return {
        "extractor": f"pdfplumber_{strategy}",
        "elapsed_seconds": perf_counter() - started_at,
        "pages": page_results,
    }


def probe_camelot(pdf_path, pages, *, flavor):
    started_at = perf_counter()
    page_results = []
    for page_number in pages:
        try:
            tables = camelot.read_pdf(
                str(pdf_path),
                pages=str(page_number),
                flavor=flavor,
                suppress_stdout=True,
            )
            extracted = []
            for table in tables:
                report = table.parsing_report
                extracted.append(
                    _table_summary(
                        table.df.values.tolist(),
                        extra={
                            "accuracy": float(report.get("accuracy", 0.0)),
                            "whitespace": float(
                                report.get("whitespace", 0.0)
                            ),
                            "order": int(report.get("order", 0)),
                        },
                    )
                )
            page_results.append(
                {
                    "page_number": page_number,
                    "table_count": len(extracted),
                    "tables": extracted,
                }
            )
        except Exception as exc:
            page_results.append(
                {
                    "page_number": page_number,
                    "table_count": 0,
                    "tables": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "extractor": f"camelot_{flavor}",
        "elapsed_seconds": perf_counter() - started_at,
        "pages": page_results,
    }


def write_markdown(output_path, results):
    lines = [
        "# PDF Table Extractor Probe",
        "",
        "人工检查基准：第 16 页无真实表格；第 27、28 页各有一张 11×6 表格。",
        "",
        "| extractor | seconds | page 16 | page 27 | page 28 |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in results:
        counts = {
            page["page_number"]: page["table_count"]
            for page in result["pages"]
        }
        lines.append(
            f"| {result['extractor']} "
            f"| {result['elapsed_seconds']:.3f} "
            f"| {counts.get(16, 0)} "
            f"| {counts.get(27, 0)} "
            f"| {counts.get(28, 0)} |"
        )

    for result in results:
        lines.extend(["", f"## {result['extractor']}", ""])
        for page in result["pages"]:
            lines.append(
                f"- page {page['page_number']}: "
                f"{page['table_count']} table(s)"
            )
            if "error" in page:
                lines.append(f"  - error: `{page['error']}`")
            for index, table in enumerate(page["tables"], start=1):
                lines.append(
                    f"  - table {index}: "
                    f"{table['row_count']}×{table['column_count']}, "
                    f"nonempty={table['nonempty_cell_count']}, "
                    f"preview={json.dumps(table['preview'], ensure_ascii=False)}"
                )

    Path(output_path).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pages", type=int, nargs="+", default=DEFAULT_PAGES)
    args = parser.parse_args()

    results = [
        probe_pdfplumber(args.pdf_path, args.pages, strategy="lines"),
        probe_pdfplumber(args.pdf_path, args.pages, strategy="text"),
        probe_camelot(args.pdf_path, args.pages, flavor="lattice"),
        probe_camelot(args.pdf_path, args.pages, flavor="stream"),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "table_extractor_probe.json"
    markdown_path = args.output_dir / "table_extractor_probe.md"
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(markdown_path, results)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"json: {json_path}")
    print(f"markdown: {markdown_path}")


if __name__ == "__main__":
    main()
