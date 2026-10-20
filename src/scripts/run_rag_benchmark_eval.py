"""读取端到端预测 JSONL，并输出按任务类型拆分的评测报告。"""

import argparse
import json
from pathlib import Path

from src.evaluation.rag_benchmark_metrics import evaluate_rag_benchmark


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES_PATH = PROJECT_ROOT / "data/eval/rag_benchmark_v1.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data/eval/runs/rag_benchmark_report.json"


def load_jsonl(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JSONL 文件不存在: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="按题型评测金融研报 RAG 的端到端输出"
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    result = evaluate_rag_benchmark(
        load_jsonl(args.cases),
        load_jsonl(args.predictions),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {key: value for key, value in result.items() if key != "per_query"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
