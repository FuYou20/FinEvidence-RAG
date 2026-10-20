import argparse
import json
from pathlib import Path

from src.embedder import qwen_embed_batch
from src.evaluation.retrieval_runner import (
    run_retrieval_cases_with_retriever,
    write_retrieval_run,
)
from src.evaluation.table_retrieval_metrics import (
    evaluate_table_retrieval,
    load_table_retrieval_cases,
    write_table_metrics_report,
)
from src.hybrid_retriever import hybrid_retrieve
from src.index_cache import load_index_bundle
from src.sparse_retriever import build_bm25_index
from src.table_parser import TABLE_EXTRACTION_VERSION


def parse_args():
    parser = argparse.ArgumentParser(description="运行表格专项检索评测")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-k", type=int, default=30)
    parser.add_argument("--sparse-k", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--dense-weight", type=float, default=4.0)
    parser.add_argument("--sparse-weight", type=float, default=1.0)
    parser.add_argument("--run-name", default="table_retrieval_baseline_v1")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.candidate_k < args.top_k:
        raise ValueError("candidate_k 不能小于 top_k")

    project_root = Path(__file__).resolve().parents[2]
    cache_dir = project_root / "data/cache/retrieval_index"
    dense_index, metadata, build_config = load_index_bundle(cache_dir)
    if build_config.get("include_tables") is not True:
        raise ValueError("当前索引未包含表格，请先重建表格索引")
    if build_config.get("table_extraction_version") != TABLE_EXTRACTION_VERSION:
        raise ValueError("当前索引的表格提取版本已过期，请先重建索引")

    cases = load_table_retrieval_cases(
        project_root / "data/eval/table_retrieval_cases_v1.jsonl"
    )
    bm25_index = build_bm25_index(metadata)

    def retrieve_fn(*, query, top_k):
        return hybrid_retrieve(
            query=query,
            dense_index=dense_index,
            bm25_index=bm25_index,
            metadata=metadata,
            embed_batch=qwen_embed_batch,
            dense_k=args.dense_k,
            sparse_k=args.sparse_k,
            top_k=top_k,
            rrf_k=args.rrf_k,
            dense_weight=args.dense_weight,
            sparse_weight=args.sparse_weight,
        )

    run_records = run_retrieval_cases_with_retriever(
        cases,
        retrieve_fn,
        candidate_k=args.candidate_k,
        page_top_k=args.top_k,
    )
    report = evaluate_table_retrieval(
        cases,
        run_records,
        top_k=args.top_k,
    )
    report.update(
        {
            "retrieval_mode": "hybrid",
            "dense_weight": args.dense_weight,
            "sparse_weight": args.sparse_weight,
            "indexed_table_chunk_count": sum(
                row.get("content_type") == "table" for row in metadata
            ),
            "mean_latency_ms": sum(
                row["latency_ms"] for row in run_records
            )
            / len(run_records),
        }
    )

    runs_dir = project_root / "data/eval/runs"
    run_path = runs_dir / f"{args.run_name}.jsonl"
    metrics_path = runs_dir / f"{args.run_name}_metrics.json"
    write_retrieval_run(run_path, run_records)
    write_table_metrics_report(metrics_path, report)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"run: {run_path}")
    print(f"metrics: {metrics_path}")


if __name__ == "__main__":
    main()
