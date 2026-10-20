import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.corpus_indexer import index_corpus, load_corpus_manifest
from src.chunk_expander import expand_neighbor_chunks
from src.embedder import qwen_embed_batch
from src.evaluation.retrieval_metrics import (
    evaluate_chunk_retrieval,
    evaluate_retrieval,
    write_metrics_report,
)
from src.evaluation.retrieval_dataset import validate_retrieval_dataset
from src.evaluation.retrieval_runner import (
    load_retrieval_cases,
    run_retrieval_cases_with_retriever,
    write_retrieval_run,
)
from src.hybrid_retriever import hybrid_retrieve
from src.index_cache import (
    compute_corpus_fingerprint,
    load_or_build_index,
)
from src.retriever import retrieve as dense_retrieve
from src.reranker import create_dashscope_rerank_batch, rerank_results
from src.sparse_retriever import build_bm25_index, sparse_retrieve
from src.table_parser import TABLE_EXTRACTION_VERSION


RETRIEVAL_MODES = ("dense", "bm25", "hybrid")


def parse_args():
    parser = argparse.ArgumentParser(
        description="运行 Dense、BM25 或 Hybrid 页面级检索评测"
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--page-top-k", type=int, default=5)
    parser.add_argument("--dense-k", type=int, default=20)
    parser.add_argument("--sparse-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--sparse-weight", type=float, default=1.0)
    parser.add_argument(
        "--bm25-k1",
        type=float,
        default=1.5,
        help="BM25 词频饱和参数",
    )
    parser.add_argument(
        "--bm25-b",
        type=float,
        default=0.75,
        help="BM25 文档长度归一化参数",
    )
    parser.add_argument(
        "--bm25-epsilon",
        type=float,
        default=0.25,
        help="rank_bm25 负 IDF 下限修正系数",
    )
    parser.add_argument(
        "--neighbor-window",
        type=int,
        default=0,
        help="相邻 chunk 扩展窗口；0 表示关闭",
    )
    parser.add_argument("--neighbor-seed-top-n", type=int, default=3)
    parser.add_argument("--neighbor-decay", type=float, default=0.8)
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="使用 Query-Document 联合打分模型重排召回候选",
    )
    parser.add_argument(
        "--rerank-model",
        default="qwen3.7-text-rerank",
    )
    parser.add_argument(
        "--include-tables",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=(*RETRIEVAL_MODES, "all"),
        default="dense",
    )
    parser.add_argument("--run-name", default="retrieval_eval_v1")
    parser.add_argument(
        "--cases-file",
        type=Path,
        default=Path("data/eval/retrieval_cases_v2.jsonl"),
        help="相对于项目根目录或绝对路径的 JSONL 评测集",
    )
    parser.add_argument(
        "--split",
        choices=("dev", "test", "all"),
        default="dev",
        help="dev 用于调参，参数冻结后再运行 test",
    )
    parser.add_argument(
        "--report-k",
        nargs="+",
        type=int,
        default=(1, 3, 5),
        help="同时报告多个 K；最大值不能超过 page-top-k/candidate-k",
    )
    return parser.parse_args()


def make_retrieve_fn(
    mode,
    *,
    dense_index,
    bm25_index,
    metadata,
    args,
    rerank_batch=None,
):
    base_retrieve_fn = None
    if mode == "dense":

        def retrieve_fn(*, query, top_k):
            return dense_retrieve(
                query=query,
                index=dense_index,
                metadata=metadata,
                embed_batch=qwen_embed_batch,
                top_k=top_k,
            )

        base_retrieve_fn = retrieve_fn

    if mode == "bm25":

        def retrieve_fn(*, query, top_k):
            return sparse_retrieve(
                query=query,
                bm25_index=bm25_index,
                metadata=metadata,
                top_k=top_k,
            )

        base_retrieve_fn = retrieve_fn

    if mode == "hybrid":

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

        base_retrieve_fn = retrieve_fn

    if base_retrieve_fn is None:
        raise ValueError(f"未知 retrieval mode: {mode}")

    if args.neighbor_window < 0:
        raise ValueError("neighbor_window 不能为负数")
    retrieve_pipeline = base_retrieve_fn

    if args.neighbor_window > 0:
        previous_retrieve_fn = retrieve_pipeline

        def expanded_retrieve_fn(
            *,
            query,
            top_k,
            _retrieve_fn=previous_retrieve_fn,
        ):
            retrieval_results = _retrieve_fn(
                query=query,
                top_k=top_k,
            )
            return expand_neighbor_chunks(
                retrieval_results,
                metadata,
                top_k=top_k,
                seed_top_n=args.neighbor_seed_top_n,
                window=args.neighbor_window,
                decay=args.neighbor_decay,
            )

        retrieve_pipeline = expanded_retrieve_fn

    if args.rerank:
        if not callable(rerank_batch):
            raise TypeError("启用 rerank 时必须提供 rerank_batch")
        previous_retrieve_fn = retrieve_pipeline

        def reranked_retrieve_fn(
            *,
            query,
            top_k,
            _retrieve_fn=previous_retrieve_fn,
        ):
            retrieval_results = _retrieve_fn(
                query=query,
                top_k=top_k,
            )
            return rerank_results(
                query=query,
                retrieval_results=retrieval_results,
                rerank_batch=rerank_batch,
                top_k=top_k,
            )

        retrieve_pipeline = reranked_retrieve_fn

    return retrieve_pipeline


def write_comparison_report(output_path, summaries):
    lines = [
        "# Retrieval Comparison",
        "",
        (
            "| mode | Page Success@K | Page Recall@K | Page nDCG@K "
            "| Chunk Success@K | Chunk Recall@K | Chunk nDCG@K "
            "| mean latency (ms) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['retrieval_mode']} "
            f"| {summary['page_success_at_k']:.4f} "
            f"| {summary['page_recall_at_k']:.4f} "
            f"| {summary['page_ndcg_at_k']:.4f} "
            f"| {summary['chunk_success_at_k']:.4f} "
            f"| {summary['chunk_recall_at_k']:.4f} "
            f"| {summary['chunk_ndcg_at_k']:.4f} "
            f"| {summary['mean_latency_ms']:.2f} |"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    model_name = os.getenv(
        "QWEN_EMBEDDING_MODEL",
        "qwen3.7-text-embedding",
    )

    corpus_records = load_corpus_manifest(
        project_root / "data/eval/corpus_v1.jsonl"
    )
    corpus_fingerprint = compute_corpus_fingerprint(
        corpus_records=corpus_records,
        project_root=project_root,
    )
    build_config = {
        "corpus_fingerprint": corpus_fingerprint,
        "embedding_model": model_name,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "include_tables": args.include_tables,
        "table_extraction_version": (
            TABLE_EXTRACTION_VERSION if args.include_tables else None
        ),
    }
    cases_path = args.cases_file
    if not cases_path.is_absolute():
        cases_path = project_root / cases_path
    all_cases = load_retrieval_cases(cases_path)

    def build_index():
        return index_corpus(
            corpus_records=corpus_records,
            project_root=project_root,
            embed_batch=qwen_embed_batch,
            model_name=model_name,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            batch_size=args.batch_size,
            include_tables=args.include_tables,
        )

    dense_index, metadata, cache_status = load_or_build_index(
        cache_dir=project_root / "data/cache/retrieval_index",
        build_config=build_config,
        build_index_fn=build_index,
    )
    print(f"index cache: {cache_status}")

    is_v2_dataset = all(
        all(field in case for field in ("category", "difficulty", "split"))
        and all(
            "source_chunk_id" in evidence
            for evidence in case["gold_evidence"]
        )
        for case in all_cases
    )
    dataset_validation = None
    if is_v2_dataset:
        dataset_validation = validate_retrieval_dataset(
            all_cases,
            metadata,
            require_disjoint_splits=True,
        )
    cases = (
        all_cases
        if args.split == "all"
        else [case for case in all_cases if case.get("split") == args.split]
    )
    if not cases:
        raise ValueError(f"评测集没有 split={args.split} 的案例")
    report_ks = sorted(set((*args.report_k, args.page_top_k)))
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in report_ks
    ):
        raise ValueError("report-k 必须全部为正整数")
    if max(report_ks) > args.page_top_k:
        raise ValueError("report-k 不能超过 page-top-k")
    if max(report_ks) > args.candidate_k:
        raise ValueError("report-k 不能超过 candidate-k")
    print(
        f"evaluation dataset: {cases_path}, split={args.split}, "
        f"cases={len(cases)}"
    )

    modes = (
        list(RETRIEVAL_MODES)
        if args.retrieval_mode == "all"
        else [args.retrieval_mode]
    )
    bm25_index = (
        build_bm25_index(
            metadata,
            k1=args.bm25_k1,
            b=args.bm25_b,
            epsilon=args.bm25_epsilon,
        )
        if any(mode in {"bm25", "hybrid"} for mode in modes)
        else None
    )
    rerank_batch = None
    if args.rerank:
        rerank_api_key = os.getenv("DASHSCOPE_API_KEY")
        if not rerank_api_key:
            raise ValueError("启用 rerank 时必须配置 DASHSCOPE_API_KEY")
        rerank_batch = create_dashscope_rerank_batch(
            api_key=rerank_api_key,
            model_name=args.rerank_model,
            endpoint=os.getenv(
                "QWEN_RERANK_ENDPOINT",
                (
                    "https://dashscope.aliyuncs.com/api/v1/services/"
                    "rerank/text-rerank/text-rerank"
                ),
            ),
        )

    runs_dir = project_root / "data/eval/runs"
    summaries = []
    for mode in modes:
        retrieve_fn = make_retrieve_fn(
            mode,
            dense_index=dense_index,
            bm25_index=bm25_index,
            metadata=metadata,
            args=args,
            rerank_batch=rerank_batch,
        )
        run_records = run_retrieval_cases_with_retriever(
            cases=cases,
            retrieve_fn=retrieve_fn,
            candidate_k=args.candidate_k,
            page_top_k=args.page_top_k,
        )
        for record in run_records:
            record["retrieval_mode"] = mode

        page_metrics_at_k = {
            str(top_k): evaluate_retrieval(
                cases,
                run_records,
                top_k=top_k,
            )
            for top_k in report_ks
        }
        chunk_metrics_at_k = (
            {
                str(top_k): evaluate_chunk_retrieval(
                    cases,
                    run_records,
                    top_k=top_k,
                )
                for top_k in report_ks
            }
            if is_v2_dataset
            else {}
        )
        primary_page_report = page_metrics_at_k[str(args.page_top_k)]
        primary_chunk_report = (
            chunk_metrics_at_k.get(str(args.page_top_k))
        )

        slice_metrics = {}
        if is_v2_dataset:
            run_by_query_id = {
                record["query_id"]: record for record in run_records
            }
            for field_name in ("category", "difficulty"):
                slice_metrics[field_name] = {}
                for field_value in sorted(
                    {case[field_name] for case in cases}
                ):
                    slice_cases = [
                        case for case in cases
                        if case[field_name] == field_value
                    ]
                    slice_runs = [
                        run_by_query_id[case["query_id"]]
                        for case in slice_cases
                    ]
                    page_slice = evaluate_retrieval(
                        slice_cases,
                        slice_runs,
                        top_k=args.page_top_k,
                    )
                    chunk_slice = evaluate_chunk_retrieval(
                        slice_cases,
                        slice_runs,
                        top_k=args.page_top_k,
                    )
                    slice_metrics[field_name][field_value] = {
                        "query_count": len(slice_cases),
                        "page": {
                            key: value for key, value in page_slice.items()
                            if key != "per_query"
                        },
                        "chunk": {
                            key: value for key, value in chunk_slice.items()
                            if key != "per_query"
                        },
                    }

        report = {
            **primary_page_report,
            "dataset": {
                "path": str(cases_path),
                "split": args.split,
                "case_count": len(cases),
                "validation": dataset_validation,
            },
            "page_metrics_at_k": page_metrics_at_k,
            "chunk_metrics_at_k": chunk_metrics_at_k,
            "slice_metrics": slice_metrics,
        }
        run_name = (
            f"{args.run_name}_{mode}"
            if len(modes) > 1
            else args.run_name
        )
        run_path = runs_dir / f"{run_name}.jsonl"
        metrics_path = runs_dir / f"{run_name}_metrics.json"
        write_retrieval_run(run_path, run_records)
        write_metrics_report(metrics_path, report)

        summary = {
            "retrieval_mode": mode,
            "split": args.split,
            "query_count": len(cases),
            "top_k": args.page_top_k,
            "page_success_at_k": primary_page_report["success_at_k"],
            "page_mrr_at_k": primary_page_report["mrr_at_k"],
            "page_recall_at_k": primary_page_report["recall_at_k"],
            "page_ndcg_at_k": primary_page_report["ndcg_at_k"],
            "chunk_success_at_k": (
                primary_chunk_report["success_at_k"]
                if primary_chunk_report else None
            ),
            "chunk_mrr_at_k": (
                primary_chunk_report["mrr_at_k"]
                if primary_chunk_report else None
            ),
            "chunk_recall_at_k": (
                primary_chunk_report["recall_at_k"]
                if primary_chunk_report else None
            ),
            "chunk_ndcg_at_k": (
                primary_chunk_report["ndcg_at_k"]
                if primary_chunk_report else None
            ),
        }
        summary["neighbor_window"] = args.neighbor_window
        summary["neighbor_seed_top_n"] = args.neighbor_seed_top_n
        summary["neighbor_decay"] = args.neighbor_decay
        summary["rerank_enabled"] = args.rerank
        summary["rerank_model"] = (
            args.rerank_model if args.rerank else None
        )
        summary["bm25_k1"] = (
            args.bm25_k1 if mode in {"bm25", "hybrid"} else None
        )
        summary["bm25_b"] = (
            args.bm25_b if mode in {"bm25", "hybrid"} else None
        )
        summary["bm25_epsilon"] = (
            args.bm25_epsilon if mode in {"bm25", "hybrid"} else None
        )
        summary["mean_expanded_neighbor_count"] = sum(
            record["expanded_neighbor_count"] for record in run_records
        ) / len(run_records)
        summary["mean_boosted_direct_count"] = sum(
            record["boosted_direct_count"] for record in run_records
        ) / len(run_records)
        summary["indexed_table_chunk_count"] = sum(
            item.get("content_type") == "table" for item in metadata
        )
        summary["mean_latency_ms"] = sum(
            record["latency_ms"] for record in run_records
        ) / len(run_records)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"run: {run_path}")
        print(f"metrics: {metrics_path}")

    if len(summaries) > 1:
        comparison_json_path = runs_dir / f"{args.run_name}_comparison.json"
        comparison_md_path = runs_dir / f"{args.run_name}_comparison.md"
        comparison_json_path.write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_comparison_report(comparison_md_path, summaries)
        print(f"comparison json: {comparison_json_path}")
        print(f"comparison markdown: {comparison_md_path}")


if __name__ == "__main__":
    main()
