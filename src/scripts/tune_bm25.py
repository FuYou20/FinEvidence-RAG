import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


def _parameter_slug(value):
    return str(value).replace("-", "m").replace(".", "p")


def parse_args():
    parser = argparse.ArgumentParser(
        description="在 dev 集上运行 BM25 k1/b 参数网格搜索"
    )
    parser.add_argument(
        "--k1-values",
        nargs="+",
        type=float,
        default=(0.6, 1.0, 1.5, 2.0),
    )
    parser.add_argument(
        "--b-values",
        nargs="+",
        type=float,
        default=(0.0, 0.5, 0.75, 1.0),
    )
    parser.add_argument("--epsilon", type=float, default=0.25)
    parser.add_argument("--split", choices=("dev",), default="dev")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--page-top-k", type=int, default=5)
    parser.add_argument("--run-prefix", default="bm25_grid_dev")
    return parser.parse_args()


def _load_summary(metrics_path, *, k1, b, epsilon):
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    page = report["page_metrics_at_k"]["5"]
    chunk = report["chunk_metrics_at_k"]["5"]
    return {
        "k1": k1,
        "b": b,
        "epsilon": epsilon,
        "page_success_at_5": page["success_at_k"],
        "page_mrr_at_5": page["mrr_at_k"],
        "page_recall_at_5": page["recall_at_k"],
        "page_ndcg_at_5": page["ndcg_at_k"],
        "chunk_success_at_5": chunk["success_at_k"],
        "chunk_mrr_at_5": chunk["mrr_at_k"],
        "chunk_recall_at_5": chunk["recall_at_k"],
        "chunk_ndcg_at_5": chunk["ndcg_at_k"],
    }


def _write_markdown(path, rows):
    lines = [
        "# BM25 Parameter Tuning (dev)",
        "",
        "排序规则：Chunk nDCG@5、Chunk Recall@5、Page nDCG@5。",
        "",
        (
            "| rank | k1 | b | epsilon | Page Recall@5 | Page nDCG@5 "
            "| Chunk Recall@5 | Chunk nDCG@5 |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"| {rank} | {row['k1']:.2f} | {row['b']:.2f} "
            f"| {row['epsilon']:.2f} "
            f"| {row['page_recall_at_5']:.4f} "
            f"| {row['page_ndcg_at_5']:.4f} "
            f"| {row['chunk_recall_at_5']:.4f} "
            f"| {row['chunk_ndcg_at_5']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    runs_dir = project_root / "data/eval/runs"
    summaries = []

    for k1, b in itertools.product(args.k1_values, args.b_values):
        run_name = (
            f"{args.run_prefix}_k1_{_parameter_slug(k1)}"
            f"_b_{_parameter_slug(b)}"
        )
        command = [
            sys.executable,
            "-m",
            "src.scripts.run_retrieval_eval",
            "--retrieval-mode",
            "bm25",
            "--split",
            args.split,
            "--candidate-k",
            str(args.candidate_k),
            "--page-top-k",
            str(args.page_top_k),
            "--bm25-k1",
            str(k1),
            "--bm25-b",
            str(b),
            "--bm25-epsilon",
            str(args.epsilon),
            "--run-name",
            run_name,
        ]
        print(f"running k1={k1}, b={b}, epsilon={args.epsilon}")
        subprocess.run(
            command,
            cwd=project_root,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        summaries.append(
            _load_summary(
                runs_dir / f"{run_name}_metrics.json",
                k1=k1,
                b=b,
                epsilon=args.epsilon,
            )
        )

    summaries.sort(
        key=lambda row: (
            -row["chunk_ndcg_at_5"],
            -row["chunk_recall_at_5"],
            -row["page_ndcg_at_5"],
            row["k1"],
            row["b"],
        )
    )
    json_path = runs_dir / f"{args.run_prefix}_summary.json"
    markdown_path = runs_dir / f"{args.run_prefix}_summary.md"
    json_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown(markdown_path, summaries)
    print(json.dumps(summaries[:5], ensure_ascii=False, indent=2))
    print(f"summary json: {json_path}")
    print(f"summary markdown: {markdown_path}")


if __name__ == "__main__":
    main()
