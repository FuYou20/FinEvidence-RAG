import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.embedder import qwen_embed_batch
from src.grounded_qa import answer_with_citations
from src.hybrid_retriever import hybrid_retrieve
from src.index_cache import load_index_bundle
from src.llm_generator import create_grounded_answer_generator
from src.sparse_retriever import build_bm25_index


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "retrieval_index"
DEFAULT_QUESTION = "传音控股在非洲市场具有哪些竞争优势？"


def parse_args():
    parser = argparse.ArgumentParser(
        description="运行带可验证来源引用的 Hybrid RAG 问答 Demo"
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help="需要依据研报证据回答的问题",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-k", type=int, default=20)
    parser.add_argument("--sparse-k", type=int, default=20)
    return parser.parse_args()


def require_env(name):
    """读取必需的非空环境变量。"""
    value = os.getenv(name)
    if not value or not value.strip():
        raise RuntimeError(
            f"未检测到 {name}，请检查 {ENV_PATH} 文件。"
        )
    return value.strip()


def init_client():
    """初始化用于生成结构化引用回答的兼容客户端。"""
    return OpenAI(
        api_key=require_env("DEEPSEEK_API_KEY"),
        base_url=require_env("DEEPSEEK_BASE_URL"),
        timeout=60.0,
        max_retries=1,
    )


def validate_args(args):
    if not isinstance(args.question, str) or not args.question.strip():
        raise ValueError("question 不能为空")
    for field_name in ("top_k", "dense_k", "sparse_k"):
        value = getattr(args, field_name)
        if value <= 0:
            raise ValueError(f"{field_name} 必须是正整数")


def print_answer(answer):
    """用适合终端阅读的格式打印结构化回答。"""
    print(f"\n回答状态：{answer.status}")

    if answer.claims:
        print("\n结论：")
        for position, claim in enumerate(answer.claims, start=1):
            source_labels = ", ".join(
                f"[{source_id}]" for source_id in claim.source_ids
            )
            print(f"{position}. {claim.claim} {source_labels}")

    if answer.missing_information:
        print("\n缺失或冲突信息：")
        for item in answer.missing_information:
            print(f"- {item}")


def print_sources(source_map):
    """打印回答可追溯到的检索来源。"""
    print("\n来源：")
    for source_id, source in source_map.items():
        source_type = source.get("content_type", "text")
        fields = [
            f"doc_id={source['doc_id']}",
            f"page={source['page_number']}",
            f"chunk_id={source['chunk_id']}",
            f"type={source_type}",
        ]
        if source_type == "table":
            fields.append(f"table_id={source['table_id']}")
        print(f"[{source_id}] " + " ".join(fields))


def main():
    args = parse_args()
    validate_args(args)
    question = args.question.strip()

    load_dotenv(ENV_PATH)
    client = init_client()
    model_name = require_env("DEEPSEEK_MODEL")

    dense_index, metadata, build_config = load_index_bundle(CACHE_DIR)
    bm25_index = build_bm25_index(metadata)
    print(
        "索引加载完成："
        f"chunks={dense_index.ntotal}, "
        f"dimension={dense_index.d}, "
        f"embedding_model={build_config.get('embedding_model')}"
    )

    retrieval_results = hybrid_retrieve(
        query=question,
        dense_index=dense_index,
        bm25_index=bm25_index,
        metadata=metadata,
        embed_batch=qwen_embed_batch,
        dense_k=args.dense_k,
        sparse_k=args.sparse_k,
        top_k=args.top_k,
    )
    if not retrieval_results:
        raise RuntimeError("Hybrid Retrieval 未召回任何证据")

    generate_answer = create_grounded_answer_generator(
        client,
        model_name=model_name,
    )
    answer, source_map = answer_with_citations(
        question=question,
        retrieval_results=retrieval_results,
        generate_answer=generate_answer,
    )

    print(f"\n问题：{question}")
    print_answer(answer)
    print_sources(source_map)


if __name__ == "__main__":
    main()
