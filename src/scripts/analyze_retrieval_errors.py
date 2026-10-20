import argparse
import json
from pathlib import Path

from src.evaluation.error_analysis import (
    analyze_retrieval_errors,
    write_error_analysis_jsonl,
    write_error_analysis_markdown,
)
from src.evaluation.retrieval_runner import load_retrieval_cases


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES_PATH = PROJECT_ROOT / "data/eval/retrieval_cases_v1.jsonl"


def load_jsonl(path):
    path = Path(path)
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path} 第 {line_number} 行不是合法 JSON"
                ) from exc
    return records


def resolve_project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args():
    parser = argparse.ArgumentParser(
        description="对检索运行结果生成结构化错误分析和 Markdown 报告"
    )
    parser.add_argument(
        "--run-path",
        required=True,
        help="检索运行 JSONL 路径，可使用项目根目录下的相对路径",
    )
    parser.add_argument(
        "--cases-path",
        default=str(DEFAULT_CASES_PATH),
        help="带 gold_evidence 的检索案例 JSONL 路径",
    )
    parser.add_argument(
        "--output-prefix",
        help="输出路径前缀；默认与 run 文件同目录",
    )
    parser.add_argument(
        "--split",
        choices=("dev", "test", "all"),
        default="all",
        help="仅分析指定 split，必须与 run 文件一致",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_path = resolve_project_path(args.run_path)
    cases_path = resolve_project_path(args.cases_path)
    if args.output_prefix:
        output_prefix = resolve_project_path(args.output_prefix)
    else:
        output_prefix = run_path.with_name(
            f"{run_path.stem}_error_analysis"
        )

    cases = load_retrieval_cases(cases_path)
    if args.split != "all":
        cases = [
            case for case in cases if case.get("split") == args.split
        ]
        if not cases:
            raise ValueError(f"评测集没有 split={args.split} 的案例")
    run_records = load_jsonl(run_path)
    analyses = analyze_retrieval_errors(cases, run_records)

    jsonl_path = output_prefix.with_suffix(".jsonl")
    markdown_path = output_prefix.with_suffix(".md")
    write_error_analysis_jsonl(jsonl_path, analyses)
    write_error_analysis_markdown(markdown_path, analyses)

    incomplete_count = sum(
        record["status"] != "complete" for record in analyses
    )
    print(f"queries: {len(analyses)}")
    print(f"incomplete: {incomplete_count}")
    print(f"jsonl: {jsonl_path}")
    print(f"markdown: {markdown_path}")


if __name__ == "__main__":
    main()
