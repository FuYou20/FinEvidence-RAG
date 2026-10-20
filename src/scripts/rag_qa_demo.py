import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

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


def print_sources(retrieval_results):
    """打印与回答引用编号一致的精简来源信息。"""
    print("\n检索来源：")
    for source_number, result in enumerate(
        retrieval_results,
        start=1,
    ):
        print(
            f"[S{source_number}] "
            f"page={result['page_number']} "
            f"chunk_id={result['chunk_id']} "
            f"type={result.get('content_type', 'text')} "
            f"score={result['score']:.4f} "
            f"source={'direct' if result.get('is_direct_hit', True) else 'neighbor'}"
        )


def main():
    load_dotenv(ENV_PATH)

    model_name = require_env("DEEPSEEK_MODEL")
    embedding_model = "qwen3.7-text-embedding"
    doc_id = "report_001"
    question = "传音控股在非洲市场具有哪些竞争优势？"
    chunk_size = 512
    overlap = 64
    top_k = 5
    # 实验开关：严格 Top-5 评测中邻居可能挤掉原始命中，因此默认关闭。
    # 需要复现实验时改为 2，并配合 neighbor_decay=0.98。
    neighbor_window = 0
    neighbor_seed_top_n = 3
    neighbor_decay = 0.98

    client = init_client()
    
    
    # 大模型接口
    generator = create_text_generator(
        client,
        model_name=model_name,
    )
    # 输入pdf地址 执行切分 向量化 建库等操作 返回按照chunks组织的原始数据和向量数据

    index, metadata = index_document(
        pdf_path=PDF_PATH,
        doc_id=doc_id,
        embed_batch=qwen_embed_batch,
        model_name=embedding_model,
        chunk_size=chunk_size,
        overlap=overlap,
        batch_size=20,
    )
    # 将提问文本向量化 并进行检索返回指定格式的结果
    retrieval_results = retrieve(
        query=question,
        index=index,
        metadata=metadata,
        embed_batch=qwen_embed_batch,
        top_k=top_k,
        neighbor_window=neighbor_window,
        neighbor_seed_top_n=neighbor_seed_top_n,
        neighbor_decay=neighbor_decay,
    )
    answer = answer_question(
        question=question,
        retrieval_results=retrieval_results,
        generate_answer=generator,
    )

    print(f"问题：{question}")
    print_sources(retrieval_results)
    print("\n回答：")
    print(answer)


if __name__ == "__main__":
    main()
