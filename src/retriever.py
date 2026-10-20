from errors import EmptyDocumentError, QueryEmbeddingError

from .chunk_expander import expand_neighbor_chunks
from .chunker import chunk_pages
from .embedder import embed_chunks
from .pdf_parser import parse_pdf
from .vector_store import build_index, search_index

def index_document(
    pdf_path,
    doc_id,
    embed_batch,
    *,
    model_name,
    chunk_size=512,
    overlap=64,
    batch_size=20,
    include_tables=True,
):
    # 切分pdf 为chunks并返回向量化结果
    parsed_doc = parse_pdf(
        pdf_path=pdf_path,
        doc_id=doc_id,
        include_tables=include_tables,
    )
    chunks = chunk_pages(
        parsed_doc=parsed_doc,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    if not chunks:
        raise EmptyDocumentError(
            f"文档没有产生可索引的文本块: doc_id={doc_id}"
        )

    embedded_chunks = embed_chunks(
        chunks=chunks,
        embed_batch=embed_batch,
        model_name=model_name,
        batch_size=batch_size,
    )
    return build_index(embedded_chunks=embedded_chunks)





def retrieve(
    query,
    index,
    metadata,
    embed_batch,
    *,
    top_k=5,
    neighbor_window=0,
    neighbor_seed_top_n=3,
    neighbor_decay=0.98,
):
    if not isinstance(query, str):
        raise TypeError("query 必须是字符串")

    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query 不能为空或只包含空白字符")

    query_vectors = embed_batch([normalized_query])
    try:
        vector_count = len(query_vectors)
    except TypeError as exc:
        raise QueryEmbeddingError(
            "Embedding Provider 必须返回向量列表"
        ) from exc

    if vector_count != 1:
        raise QueryEmbeddingError(
            "一次查询必须得到且只能得到一个向量，"
            f"实际得到 {vector_count} 个"
        )

    retrieval_results = search_index(
        index=index,
        metadata=metadata,
        query_vector=query_vectors[0],
        top_k=top_k,
    )

    if not isinstance(neighbor_window, int) or isinstance(
        neighbor_window, bool
    ):
        raise TypeError("neighbor_window 必须是整数")
    if neighbor_window < 0:
        raise ValueError("neighbor_window 不能为负数")
    if neighbor_window == 0:
        return retrieval_results

    return expand_neighbor_chunks(
        retrieval_results,
        metadata,
        top_k=top_k,
        seed_top_n=neighbor_seed_top_n,
        window=neighbor_window,
        decay=neighbor_decay,
    )
