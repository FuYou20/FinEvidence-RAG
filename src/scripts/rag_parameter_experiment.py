import argparse
import os
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from openai import OpenAI

from src.context_builder import build_context
from src.embedder import qwen_embed_batch
from src.llm_generator import create_text_generator
from src.rag_qa import answer_question
from src.retriever import index_document, retrieve


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "pdf"
    / "传音控股--奔跑的新兴市场，传音的长期主义--2022-04-22.pdf"
)
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"

CHUNK_SIZES = [256, 512, 1024]
TOP_K_VALUES = [3, 5, 10]


def require_env(name):
    """读取一个必需的环境变量，并拒绝空白值。"""
    value = os.getenv(name)
    if not value or not value.strip():
        raise RuntimeError(
            f"未检测到 {name}，请检查 {ENV_PATH} 文件。"
        )
    return value.strip()


def init_client():
    """使用已经加载的环境变量初始化文本生成客户端。"""
    return OpenAI(
        api_key=require_env("DEEPSEEK_API_KEY"),
        base_url=require_env("DEEPSEEK_BASE_URL"),
        timeout=45.0,
        max_retries=1,
    )


def summarize_sources(retrieval_results):
    """只保留实验分析所需的检索元数据。"""
    return [
        {
            "rank": result["rank"],
            "chunk_id": result["chunk_id"],
            "page_number": result["page_number"],
            "score": result["score"],
        }
        for result in retrieval_results
    ]


def run_experiments(
    *,
    chunk_sizes,
    top_k_values,
    generator,
    question,
):
    """为每个 chunk_size 建一次索引，并运行所有 top-k 组合。"""
    results = []
    embedding_model = "qwen3.7-text-embedding"
    doc_id = "report_001"

    for chunk_size in chunk_sizes:
        print(f"\n正在建立索引：chunk_size={chunk_size}")
        index, metadata = index_document(
            pdf_path=PDF_PATH,
            doc_id=doc_id,
            embed_batch=qwen_embed_batch,
            model_name=embedding_model,
            chunk_size=chunk_size,
            overlap=64,
            batch_size=20,
        )

        for top_k in top_k_values:
            print(
                "正在运行问答："
                f"chunk_size={chunk_size}, top_k={top_k}"
            )
            started_at = perf_counter()
            retrieval_results = retrieve(
                query=question,
                index=index,
                metadata=metadata,
                embed_batch=qwen_embed_batch,
                top_k=top_k,
            )
            context = build_context(retrieval_results)
            answer = answer_question(
                question=question,
                retrieval_results=retrieval_results,
                generate_answer=generator,
            )
            latency_seconds = perf_counter() - started_at

            results.append(
                {
                    "chunk_size": chunk_size,
                    "top_k": top_k,
                    "total_chunks": len(metadata),
                    "retrieved_sources": summarize_sources(
                        retrieval_results
                    ),
                    "context_chars": len(context),
                    "answer": answer,
                    "latency_seconds": latency_seconds,
                }
            )

    return results


def write_markdown_report(*, results, question, output_path):
    """把原始实验数据写成便于人工评分的 Markdown 报告。"""
    lines = [
        "# RAG Parameter Experiment",
        "",
        f"**Question:** {question}",
        "",
        (
            "| chunk_size | top_k | total_chunks | context_chars "
            "| latency_seconds | quality_score |"
        ),
        "|---:|---:|---:|---:|---:|---:|",
    ]

    for result in results:
        lines.append(
            f"| {result['chunk_size']} | {result['top_k']} "
            f"| {result['total_chunks']} "
            f"| {result['context_chars']} "
            f"| {result['latency_seconds']:.2f} | 待人工评分/7 |"
        )

    for result in results:
        chunk_size = result["chunk_size"]
        top_k = result["top_k"]
        lines.extend(
            [
                "",
                f"## chunk_size={chunk_size}, top_k={top_k}",
                "",
                "### Retrieved Sources",
                "",
            ]
        )
        for source_number, source in enumerate(
            result["retrieved_sources"],
            start=1,
        ):
            lines.append(
                f"- [S{source_number}] "
                f"rank={source['rank']}, "
                f"page={source['page_number']}, "
                f"chunk_id={source['chunk_id']}, "
                f"score={source['score']:.4f}"
            )
        lines.extend(
            [
                "",
                "### Answer",
                "",
                result["answer"],
                "",
                "### Manual Evaluation",
                "",
                "- Quality score: 待评分/7",
                "- Observation: 待填写",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="比较 RAG 的 chunk_size 与 top_k 参数。"
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="只运行 chunk_size=256、top_k=3 的冒烟实验。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    load_dotenv(ENV_PATH)

    question = "传音控股在非洲市场具有哪些竞争优势？"
    client = init_client()
    generator = create_text_generator(
        client,
        model_name=require_env("DEEPSEEK_MODEL"),
    )

    if args.single:
        chunk_sizes = CHUNK_SIZES[:1]
        top_k_values = TOP_K_VALUES[:1]
        output_path = (
            EXPERIMENT_DIR / "rag_parameter_experiment_single.md"
        )
    else:
        chunk_sizes = CHUNK_SIZES
        top_k_values = TOP_K_VALUES
        output_path = EXPERIMENT_DIR / "rag_parameter_experiment.md"

    results = run_experiments(
        chunk_sizes=chunk_sizes,
        top_k_values=top_k_values,
        generator=generator,
        question=question,
    )
    write_markdown_report(
        results=results,
        question=question,
        output_path=output_path,
    )
    print(f"\n实验完成，共 {len(results)} 组。")
    print(f"结果已保存：{output_path}")


if __name__ == "__main__":
    main()
