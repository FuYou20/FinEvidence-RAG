"""汇总单事实与枚举任务的端到端消融实验。"""

import argparse
import json
from pathlib import Path


METRICS = (
    ("single_fact", "accuracy", "Single Accuracy"),
    ("single_fact", "fact_recall", "Single Fact Recall"),
    ("single_fact", "evidence_chunk_recall", "Single Evidence Recall"),
    ("enumeration", "item_precision", "Enum Precision"),
    ("enumeration", "item_recall", "Enum Recall"),
    ("enumeration", "item_f1", "Enum F1"),
    ("enumeration", "complete_answer", "Enum Complete Rate"),
    ("enumeration", "item_citation_accuracy", "Enum Citation Accuracy"),
    ("enumeration", "evidence_chunk_recall", "Enum Evidence Recall"),
)


def parse_args():
    parser = argparse.ArgumentParser(description="汇总RAG任务级消融结果")
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        metavar="NAME=REPORT_JSON",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_variants(specifications):
    variants = []
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(f"variant 必须是 NAME=PATH: {specification}")
        name, raw_path = specification.split("=", 1)
        path = Path(raw_path)
        if not name.strip() or not path.is_file():
            raise ValueError(f"无效 variant: {specification}")
        report = json.loads(path.read_text(encoding="utf-8"))
        variants.append((name.strip(), report))
    return variants


def main():
    args = parse_args()
    variants = load_variants(args.variant)
    baseline = variants[0][1]

    lines = [
        "# Single-Fact / Enumeration Ablation",
        "",
        "| variant | " + " | ".join(label for _, _, label in METRICS) + " |",
        "|---|" + "---:|" * len(METRICS),
    ]
    for name, report in variants:
        values = [report[section][metric] for section, metric, _ in METRICS]
        lines.append(
            f"| {name} | " + " | ".join(f"{value:.4f}" for value in values) + " |"
        )

    lines.extend(["", "## Delta vs Dense", ""])
    for name, report in variants[1:]:
        lines.append(f"### {name}")
        lines.append("")
        for section, metric, label in METRICS:
            delta = report[section][metric] - baseline[section][metric]
            lines.append(f"- {label}: {delta:+.4f}")
        lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
