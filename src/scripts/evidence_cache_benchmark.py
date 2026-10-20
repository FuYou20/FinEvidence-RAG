"""Run a reproducible no-cache/cold-cache/warm-cache multi-agent benchmark."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from uuid import uuid4

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOPIC = "传音控股在非洲市场的竞争优势是否具有持续性？"


def parse_args():
    parser = argparse.ArgumentParser(
        description="比较无缓存、冷缓存和热缓存的真实多 Agent 运行"
    )
    parser.add_argument("topic", nargs="?", default=DEFAULT_TOPIC)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="实验输出目录；默认写入 data/eval/runs",
    )
    return parser.parse_args()


def _run_case(*, label, topic, output_dir, env, disable_cache):
    output_path = output_dir / f"{label}.json"
    command = [
        sys.executable,
        "-m",
        "src.scripts.multi_agent_demo",
        topic,
        "--max-rounds", "1",
        "--max-challenges", "0",
        "--max-verifications", "0",
        "--max-deepening-steps", "1",
        "--disable-web",
        "--no-memory",
        "--no-color",
        "--output", str(output_path),
    ]
    if disable_cache:
        command.append("--disable-evidence-cache")
    started = perf_counter()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    elapsed = round(perf_counter() - started, 6)
    (output_dir / f"{label}.log").write_text(
        completed.stdout + "\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} 运行失败，详见 {output_dir / f'{label}.log'}"
        )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    performance = payload["performance"]
    runtime = performance["evidence_runtime"]
    hits = runtime.get("evidence_cache_hits", 0)
    misses = runtime.get("evidence_cache_misses", 0)
    attempts = hits + misses
    return {
        "label": label,
        "output_path": str(output_path),
        "process_wall_time_seconds": elapsed,
        "pipeline_wall_time_seconds": performance["wall_time_seconds"],
        "model_calls": performance["model"]["total_calls"],
        "model_input_tokens": performance["model"].get("input_tokens"),
        "model_total_tokens": performance["model"].get("total_tokens"),
        "embedding_calls": performance["external_calls"]["embedding"]["total_calls"],
        "cache_hits": hits,
        "cache_misses": misses,
        "cache_hit_rate": round(hits / attempts, 6) if attempts else None,
        "cache_post_lock_hits": runtime.get(
            "evidence_cache_post_lock_hits", 0
        ),
        "cache_lock_waits": runtime.get("evidence_cache_lock_waits", 0),
        "avoided_investigations": runtime.get(
            "avoided_investigations", 0
        ),
        "evidence_requests": runtime.get("evidence_requests", 0),
    }


def _difference(baseline, warm):
    baseline_seconds = baseline["pipeline_wall_time_seconds"]
    warm_seconds = warm["pipeline_wall_time_seconds"]
    saved_seconds = round(baseline_seconds - warm_seconds, 6)
    return {
        "saved_model_calls": baseline["model_calls"] - warm["model_calls"],
        "saved_embedding_calls": (
            baseline["embedding_calls"] - warm["embedding_calls"]
        ),
        "saved_input_tokens": (
            baseline["model_input_tokens"] - warm["model_input_tokens"]
            if baseline["model_input_tokens"] is not None
            and warm["model_input_tokens"] is not None
            else None
        ),
        "saved_total_tokens": (
            baseline["model_total_tokens"] - warm["model_total_tokens"]
            if baseline["model_total_tokens"] is not None
            and warm["model_total_tokens"] is not None
            else None
        ),
        "saved_investigations": warm["avoided_investigations"],
        "saved_pipeline_seconds": saved_seconds,
        "pipeline_latency_reduction": (
            round(saved_seconds / baseline_seconds, 6)
            if baseline_seconds > 0
            else None
        ),
        "warm_cache_hit_rate": warm["cache_hit_rate"],
    }


def _write_markdown(summary, path):
    cases = summary["cases"]
    comparison = summary["comparison_no_cache_vs_warm"]
    rows = [
        "# Redis 证据缓存真实多 Agent 对比实验",
        "",
        f"- 议题：{summary['topic']}",
        f"- 实验时间：{summary['created_at']}",
        "- 固定配置：1 轮辩论、0 条质疑、0 次裁判复核、禁用联网",
        "",
        "| 组别 | 模型调用 | Embedding 调用 | 缓存命中率 | 避免调查 | 流水线耗时(s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in cases:
        hit_rate = (
            "—" if item["cache_hit_rate"] is None
            else f"{item['cache_hit_rate']:.1%}"
        )
        rows.append(
            f"| {item['label']} | {item['model_calls']} | "
            f"{item['embedding_calls']} | {hit_rate} | "
            f"{item['avoided_investigations']} | "
            f"{item['pipeline_wall_time_seconds']:.3f} |"
        )
    reduction = comparison["pipeline_latency_reduction"]
    rows.extend([
        "",
        "## 无缓存 vs 热缓存",
        "",
        f"- 节省模型调用：{comparison['saved_model_calls']}",
        f"- 节省 Embedding 调用：{comparison['saved_embedding_calls']}",
        f"- 避免重复调查：{comparison['saved_investigations']}",
        f"- 节省响应时间：{comparison['saved_pipeline_seconds']:.3f} 秒",
        f"- 响应时间下降：{reduction:.1%}" if reduction is not None else "- 响应时间下降：—",
    ])
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("REDIS_URL"):
        raise RuntimeError("缺少 REDIS_URL，无法运行缓存实验")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir
        or PROJECT_ROOT / "data" / "eval" / "runs" / f"cache_ab_{timestamp}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # A unique namespace makes every benchmark start from a guaranteed cold
    # cache without deleting unrelated Redis keys.
    env["EVIDENCE_CACHE_NAMESPACE_SALT"] = f"benchmark-{uuid4().hex}"
    cases = []
    experiment_plan = (
        ("no_cache", True),
        ("cold_cache", False),
        ("warm_cache", False),
    )
    for label, disable_cache in experiment_plan:
        print(f"[{label}] 开始真实多 Agent 运行", flush=True)
        result = _run_case(
            label=label,
            topic=args.topic.strip(),
            output_dir=output_dir,
            env=env,
            disable_cache=disable_cache,
        )
        cases.append(result)
        print(
            f"[{label}] 完成: model_calls={result['model_calls']}, "
            f"cache_hits={result['cache_hits']}, "
            f"time={result['pipeline_wall_time_seconds']:.3f}s",
            flush=True,
        )
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "topic": args.topic.strip(),
        "cases": cases,
        "comparison_no_cache_vs_warm": _difference(cases[0], cases[2]),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown(summary, output_dir / "summary.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"实验结果: {summary_path}")


if __name__ == "__main__":
    main()
