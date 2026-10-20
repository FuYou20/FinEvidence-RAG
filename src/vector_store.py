
import numpy as np
import faiss
import copy

def build_index(embedded_chunks):
    if embedded_chunks is None or len(embedded_chunks) == 0:
        raise ValueError("embeded_chunks")
    vector_list = [
    chunk["embedding"]
    for chunk in embedded_chunks
]
    vectors_matrix = np.array(vector_list, dtype=np.float32)   # 必须是 float32
    if vectors_matrix.ndim != 2:
        raise ValueError("vectors_matrix.ndim != 2")
   
    num, dim = vectors_matrix.shape
    if num != len(embedded_chunks):
        raise ValueError("num != len(embedded_chunks)")
    if dim <= 0:
        raise ValueError("dim <= 0")
    # 归一化之前拒绝零向量
    # 每行向量对应的模
    
    norms = np.linalg.norm(vectors_matrix, axis=1)
    if 0 in norms:
        raise ValueError("norm = 0")
    # 归一化
    faiss.normalize_L2(vectors_matrix)
    # COS 距离
    index = faiss.IndexFlatIP(dim)
    index.add(vectors_matrix)
    
    # 构建metadata
    metadata = [
    {
        key: value
        for key, value in chunk.items()
        if key != "embedding"
    }
    for chunk in embedded_chunks
]
    
    return index,metadata



def search_index(index, metadata, query_vector, top_k=5):
    if index is None:
        raise ValueError
    if index.ntotal==0:
        raise ValueError
    if len(metadata) != index.ntotal:
        raise ValueError
    
    if not isinstance(top_k,int):
        raise ValueError
    if top_k<=0:
        raise ValueError
    query_array = np.array(
    query_vector,
    dtype=np.float32,
    copy=True,
)
    if query_array.ndim != 1:
        raise ValueError("query_vector 必须是一维向量")
    
    query_dim = query_array.shape[0]
    # 固定为 1 行，列数根据向量长度自动计算
    query_matrix = query_array.reshape(1, -1)
    if index.d != query_dim:
        raise ValueError
    norm = np.linalg.norm(query_array)

    if norm == 0:
        raise ValueError("查询向量存在零向量，无法归一化")
    # 归一化
    faiss.normalize_L2(query_matrix)
    effective_k = min(top_k,index.ntotal)
    scores, positions = index.search(query_matrix, effective_k)
    

    
    
    
    query_result = []
    for rank in range(effective_k):
        idx = int(positions[0][rank])

        # FAISS 返回 -1 表示该位置没有有效结果
        if idx == -1:
            continue
        metadata_copy = copy.deepcopy(metadata[idx])
        metadata_copy.update(
    {
        "index_id": idx,
        "rank": rank + 1,
        "score": float(scores[0][rank]),
    }
)
        query_result.append(metadata_copy)


    return query_result