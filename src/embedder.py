import os
from pathlib import Path

from dotenv import load_dotenv

from .qwen_embedding_client import init_qwen_embedding_client

project_root = Path(__file__).resolve().parents[1]
load_dotenv(project_root / ".env")


qwen_embed_client = None


def _get_qwen_embed_client():
    """Lazily create the provider client when a real request is made.

    Keeping provider configuration out of module import makes the core
    chunking/retrieval package usable in tests and offline workflows.
    """
    global qwen_embed_client
    if qwen_embed_client is not None:
        return qwen_embed_client

    api_key = os.getenv("DASHSCOPE_API_KEY")
    embedding_base_url = os.getenv("QWEN_EMBEDDING_BASE_URL")
    if not api_key:
        raise RuntimeError("未读取到 DASHSCOPE_API_KEY")
    if not embedding_base_url:
        raise RuntimeError("未读取到 QWEN_EMBEDDING_BASE_URL")

    qwen_embed_client = init_qwen_embedding_client(
        base_url=embedding_base_url,
        api_key=api_key,
    )
    return qwen_embed_client


def fake_embed_batch(texts):
    return [
        [float(len(text)), 1.0, 2.0]
        for text in texts
    ]
def qwen_embed_batch(texts: list[str],model_name="qwen3.7-text-embedding") -> list[list[float]]:
    if not texts:
            return []
    client = _get_qwen_embed_client()
    response = client.embeddings.create(
    model=model_name,
    input=texts,
)
    
    vectors = [item.embedding for item in response.data]
    if len(texts) != len(vectors):
        raise ValueError("embedding模型返回向量数目错误")
        
    return vectors
    


def embed_chunks(
    chunks,
    embed_batch,
    *,
    model_name,
    batch_size=32,
):
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size 必须是正整数")

    if not callable(embed_batch):
        raise TypeError("embed_batch 必须是可调用对象")

    if not chunks:
        return []

    results = []
    expected_dimension = None

    for start in range(0, len(chunks), batch_size):
        batch_chunks = chunks[start:start + batch_size]

        texts = [
            chunk["text"]
            for chunk in batch_chunks
        ]

        vectors = embed_batch(texts)

        if len(vectors) != len(texts):
            raise ValueError("返回向量数量与输入文本数量不一致")

        for chunk, vector in zip(batch_chunks, vectors):
            if len(vector) == 0:
                raise ValueError("向量维度为0")
            if expected_dimension is None:
                expected_dimension = len(vector)
            elif len(vector) != expected_dimension:
                raise ValueError("向量长度不统一")
            embedded_chunk = {
                    **chunk,
                    "embedding": list(vector),
                    "embedding_model": model_name,
                    "embedding_dimension": len(vector),
                }
            results.append(embedded_chunk)
            

    return results
    
        
    

    
