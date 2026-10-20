"""运行50题金融研报端到端 RAG 基准并持续写入可恢复检查点。"""

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.chunk_expander import expand_neighbor_chunks
from src.embedder import qwen_embed_batch
from src.evaluation.rag_benchmark_generation import (
    create_benchmark_answer_generator,
    create_comparison_judge,
    create_enumeration_judge,
    generate_prediction_record,
)
from src.evaluation.rag_benchmark_metrics import evaluate_rag_benchmark
from src.index_cache import load_index_bundle
from src.performance_monitor import InstrumentedClient, ModelCallMonitor
from src.reranker import create_dashscope_rerank_batch, rerank_results
from src.retriever import retrieve as dense_retrieve


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_CASES_PATH = PROJECT_ROOT / "data/eval/rag_benchmark_v1.jsonl"
DEFAULT_PREDICTIONS_PATH = (
    PROJECT_ROOT / "data/eval/runs/rag_benchmark_v1_predictions.jsonl"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "data/eval/runs/rag_benchmark_v1_report.json"
)
DEFAULT_RUNTIME_PATH = (
    PROJECT_ROOT / "data/eval/runs/rag_benchmark_v1_runtime.json"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="运行端到端金融研报 RAG 基准"
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument(
        "--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME_PATH)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--neighbor-window", type=int, default=1)
    parser.add_argument("--neighbor-seed-top-n", type=int, default=3)
    parser.add_argument("--neighbor-decay", type=float, default=0.8)
    parser.add_argument("--rerank-model", default="qwen3.7-text-rerank")
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--model-name")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--judge-comparisons",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--judge-enumerations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--force-query-id",
        action="append",
        default=[],
        help="忽略这些 query_id 的已有检查点并重新生成",
    )
    return parser.parse_args()


def load_jsonl(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JSONL 文件不存在: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def write_jsonl(path, records):
    """原子改写检查点，供第二阶段追加裁判结果。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())
    temporary_path.replace(path)


def require_env(name):
    value = os.getenv(name)
    if not value or not value.strip():
        raise RuntimeError(f"未检测到 {name}，请检查 {ENV_PATH}")
    return value.strip()


def validate_args(args):
    for field_name in (
        "candidate_k",
        "top_k",
        "neighbor_window",
        "neighbor_seed_top_n",
        "max_attempts",
    ):
        value = getattr(args, field_name)
        minimum = 0 if field_name == "neighbor_window" else 1
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{field_name} 必须大于等于 {minimum}")
    if args.top_k > args.candidate_k:
        raise ValueError("top_k 不能超过 candidate_k")
    if not 0 < args.neighbor_decay <= 1:
        raise ValueError("neighbor_decay 必须满足 0 < value <= 1")
    if args.max_cases is not None and args.max_cases <= 0:
        raise ValueError("max_cases 必须是正整数")


def retrieve_evidence(
    *,
    query,
    dense_index,
    metadata,
    rerank_batch,
    args,
):
    candidates = dense_retrieve(
        query=query,
        index=dense_index,
        metadata=metadata,
        embed_batch=qwen_embed_batch,
        top_k=args.candidate_k,
    )
    if args.neighbor_window:
        candidates = expand_neighbor_chunks(
            candidates,
            metadata,
            top_k=args.candidate_k,
            seed_top_n=args.neighbor_seed_top_n,
            window=args.neighbor_window,
            decay=args.neighbor_decay,
        )
    if args.rerank:
        return rerank_results(
            query=query,
            retrieval_results=candidates,
            rerank_batch=rerank_batch,
            top_k=args.top_k,
        )
    results = candidates[: args.top_k]
    for rank, item in enumerate(results, start=1):
        item["rank"] = rank
    return results


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    validate_args(args)
    load_dotenv(ENV_PATH)

    all_cases = load_jsonl(args.cases)
    cases = (
        all_cases
        if args.split == "all"
        else [case for case in all_cases if case["split"] == args.split]
    )
    if args.max_cases is not None:
        cases = cases[:args.max_cases]
    if not cases:
        raise ValueError("没有待运行的评测案例")

    if not args.resume and args.predictions.exists():
        args.predictions.unlink()
    existing_records = (
        load_jsonl(args.predictions)
        if args.resume and args.predictions.is_file()
        else []
    )
    existing_by_id = {record["query_id"]: record for record in existing_records}
    if len(existing_by_id) != len(existing_records):
        raise ValueError("已有预测文件包含重复 query_id")
    force_query_ids = set(args.force_query_id)
    unknown_force_ids = force_query_ids - {case["query_id"] for case in cases}
    if unknown_force_ids:
        raise ValueError(f"force-query-id 不在当前案例中: {sorted(unknown_force_ids)}")
    if force_query_ids:
        for query_id in force_query_ids:
            existing_by_id.pop(query_id, None)
        write_jsonl(
            args.predictions,
            [
                record
                for case in cases
                if (record := existing_by_id.get(case["query_id"])) is not None
            ],
        )

    dense_index, metadata, build_config = load_index_bundle(
        PROJECT_ROOT / "data/cache/retrieval_index"
    )
    rerank_batch = None
    if args.rerank:
        rerank_batch = create_dashscope_rerank_batch(
            api_key=require_env("DASHSCOPE_API_KEY"),
            model_name=args.rerank_model,
            endpoint=os.getenv("QWEN_RERANK_ENDPOINT")
            or (
                "https://dashscope.aliyuncs.com/api/v1/services/"
                "rerank/text-rerank/text-rerank"
            ),
        )

    model_name = (args.model_name or require_env("DEEPSEEK_MODEL")).strip()
    monitor = ModelCallMonitor()
    client = InstrumentedClient(
        OpenAI(
            api_key=require_env("DEEPSEEK_API_KEY"),
            base_url=require_env("DEEPSEEK_BASE_URL"),
            timeout=90.0,
            max_retries=2,
        ),
        monitor,
    )
    generate_answer = create_benchmark_answer_generator(
        client, model_name=model_name
    )
    comparison_judge = (
        create_comparison_judge(client, model_name=model_name)
        if args.judge_comparisons
        else None
    )
    enumeration_judge = (
        create_enumeration_judge(client, model_name=model_name)
        if args.judge_enumerations
        else None
    )

    started = time.perf_counter()
    case_runtime = []
    total = len(cases)
    for position, case in enumerate(cases, start=1):
        query_id = case["query_id"]
        if query_id in existing_by_id:
            print(f"[{position}/{total}] skip {query_id} (checkpoint)", flush=True)
            continue

        for attempt in range(1, args.max_attempts + 1):
            case_started = time.perf_counter()
            try:
                retrieval_started = time.perf_counter()
                retrieval_results = retrieve_evidence(
                    query=case["question"],
                    dense_index=dense_index,
                    metadata=metadata,
                    rerank_batch=rerank_batch,
                    args=args,
                )
                retrieval_seconds = time.perf_counter() - retrieval_started
                record = generate_prediction_record(
                    case=case,
                    retrieval_results=retrieval_results,
                    generate_answer=generate_answer,
                    # 回答生成和语义裁判分成两个可恢复阶段，避免裁判失败时
                    # 重复进行检索和回答调用。
                    comparison_judge=None,
                )
                record["run_metadata"] = {
                    "model_name": model_name,
                    "embedding_model": build_config.get("embedding_model"),
                    "retrieval": "_".join(
                        [
                            "dense",
                            *(["neighbor"] if args.neighbor_window else []),
                            *(["rerank"] if args.rerank else []),
                        ]
                    ),
                    "candidate_k": args.candidate_k,
                    "top_k": args.top_k,
                    "neighbor_window": args.neighbor_window,
                    "rerank_model": args.rerank_model if args.rerank else None,
                }
                append_jsonl(args.predictions, record)
                existing_by_id[query_id] = record
                duration = time.perf_counter() - case_started
                case_runtime.append(
                    {
                        "query_id": query_id,
                        "attempt": attempt,
                        "retrieval_seconds": round(retrieval_seconds, 6),
                        "total_seconds": round(duration, 6),
                    }
                )
                print(
                    f"[{position}/{total}] done {query_id} "
                    f"type={case['task_type']} action={record['predicted_action']} "
                    f"seconds={duration:.2f}",
                    flush=True,
                )
                break
            except Exception as exc:
                print(
                    f"[{position}/{total}] failed {query_id} "
                    f"attempt={attempt}/{args.max_attempts} "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                if attempt == args.max_attempts:
                    raise

    enumeration_judge_runtime = []
    if enumeration_judge is not None:
        enumeration_cases = [
            case for case in cases if case["task_type"] == "enumeration"
        ]
        for position, case in enumerate(enumeration_cases, start=1):
            record = existing_by_id[case["query_id"]]
            if record.get("enumeration_matches") is not None:
                print(
                    f"[enum judge {position}/{len(enumeration_cases)}] "
                    f"skip {case['query_id']} (checkpoint)",
                    flush=True,
                )
                continue
            gold_items = [item["item"] for item in case["gold_items"]]
            predicted_items = [
                item["item"] for item in record["predicted_items"]
            ]
            if not predicted_items:
                record["enumeration_matches"] = []
                record["enumeration_judgment"] = {
                    "matches": [],
                    "rationale": "预测没有枚举项目",
                }
                write_jsonl(
                    args.predictions,
                    [existing_by_id[item["query_id"]] for item in cases],
                )
                continue
            for attempt in range(1, args.max_attempts + 1):
                judge_started = time.perf_counter()
                try:
                    judgment = enumeration_judge(
                        gold_items=gold_items,
                        predicted_items=predicted_items,
                    )
                    record["enumeration_matches"] = [
                        match.model_dump() for match in judgment.matches
                    ]
                    record["enumeration_judgment"] = judgment.model_dump()
                    write_jsonl(
                        args.predictions,
                        [existing_by_id[item["query_id"]] for item in cases],
                    )
                    duration = time.perf_counter() - judge_started
                    enumeration_judge_runtime.append(
                        {
                            "query_id": case["query_id"],
                            "attempt": attempt,
                            "seconds": round(duration, 6),
                        }
                    )
                    print(
                        f"[enum judge {position}/{len(enumeration_cases)}] "
                        f"done {case['query_id']} "
                        f"matches={len(judgment.matches)} "
                        f"seconds={duration:.2f}",
                        flush=True,
                    )
                    break
                except Exception as exc:
                    print(
                        f"[enum judge {position}/{len(enumeration_cases)}] "
                        f"failed {case['query_id']} "
                        f"attempt={attempt}/{args.max_attempts} "
                        f"error={type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    if attempt == args.max_attempts:
                        raise

    judge_runtime = []
    if comparison_judge is not None:
        comparison_cases = [
            case for case in cases if case["task_type"] == "comparison"
        ]
        for position, case in enumerate(comparison_cases, start=1):
            record = existing_by_id[case["query_id"]]
            if record.get("comparison_score") is not None:
                print(
                    f"[judge {position}/{len(comparison_cases)}] "
                    f"skip {case['query_id']} (checkpoint)",
                    flush=True,
                )
                continue
            for attempt in range(1, args.max_attempts + 1):
                judge_started = time.perf_counter()
                try:
                    judgment = comparison_judge(
                        question=case["question"],
                        reference_answer=case["reference_answer"],
                        candidate_answer=record["answer_text"],
                    )
                    record["comparison_score"] = judgment.overall_score
                    record["comparison_judgment"] = judgment.model_dump()
                    write_jsonl(
                        args.predictions,
                        [existing_by_id[item["query_id"]] for item in cases],
                    )
                    duration = time.perf_counter() - judge_started
                    judge_runtime.append(
                        {
                            "query_id": case["query_id"],
                            "attempt": attempt,
                            "seconds": round(duration, 6),
                        }
                    )
                    print(
                        f"[judge {position}/{len(comparison_cases)}] "
                        f"done {case['query_id']} "
                        f"score={judgment.overall_score:.3f} "
                        f"seconds={duration:.2f}",
                        flush=True,
                    )
                    break
                except Exception as exc:
                    print(
                        f"[judge {position}/{len(comparison_cases)}] "
                        f"failed {case['query_id']} "
                        f"attempt={attempt}/{args.max_attempts} "
                        f"error={type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    if attempt == args.max_attempts:
                        raise

    selected_predictions = [existing_by_id[case["query_id"]] for case in cases]
    report = evaluate_rag_benchmark(cases, selected_predictions)
    write_json(args.report, report)
    runtime = {
        "model_name": model_name,
        "query_count": len(cases),
        "wall_seconds_this_run": round(time.perf_counter() - started, 6),
        "model_calls": monitor.snapshot(),
        "cases_completed_this_run": case_runtime,
        "enumerations_judged_this_run": enumeration_judge_runtime,
        "comparisons_judged_this_run": judge_runtime,
    }
    write_json(args.runtime, runtime)

    summary = {key: value for key, value in report.items() if key != "per_query"}
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"predictions: {args.predictions}", flush=True)
    print(f"report: {args.report}", flush=True)
    print(f"runtime: {args.runtime}", flush=True)


if __name__ == "__main__":
    main()
