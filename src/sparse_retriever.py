import copy

import jieba
from rank_bm25 import BM25Okapi


PUNCTUATION_TOKENS = frozenset(
    "，。！？、；：\"'（）《》【】,.!?;:()[]{}"
)


def tokenize_text(text):
    """将中英文混合文本切分为供 BM25 使用的规范化 token。"""
    if not isinstance(text, str):
        raise TypeError(f"期待 text 是 str 类型，实际为 {type(text)}")

    normalized_text = text.strip()
    if not normalized_text:
        raise ValueError("text 不应为空字符串")

    return [
        token.lower()
        for token in jieba.lcut(normalized_text)
        if token.strip() and token not in PUNCTUATION_TOKENS
    ]


def build_bm25_index(
    metadata,
    tokenizer=tokenize_text,
    *,
    k1=1.5,
    b=0.75,
    epsilon=0.25,
):
    """根据 chunk metadata 中的文本建立可调参数的 BM25 索引。

    ``k1`` 控制词频饱和速度，``b`` 控制文档长度归一化强度，
    ``epsilon`` 是 rank_bm25 对负 IDF 的下限修正系数。
    """
    if not isinstance(metadata, list):
        raise TypeError(f"期待 metadata 是 list，实际为 {type(metadata)}")
    if not metadata:
        raise ValueError("metadata 不应为空列表")
    if not callable(tokenizer):
        raise TypeError("tokenizer 必须是可调用对象")
    for parameter_name, parameter_value in (
        ("k1", k1),
        ("b", b),
        ("epsilon", epsilon),
    ):
        if not isinstance(parameter_value, (int, float)) or isinstance(
            parameter_value,
            bool,
        ):
            raise TypeError(f"{parameter_name} 必须是数值")
    if k1 <= 0:
        raise ValueError("k1 必须大于 0")
    if not 0 <= b <= 1:
        raise ValueError("b 必须位于 [0, 1]")
    if epsilon < 0:
        raise ValueError("epsilon 不能为负数")

    tokenized_corpus = []
    for row_number, row_data in enumerate(metadata, start=1):
        if not isinstance(row_data, dict):
            raise TypeError(
                f"期待第 {row_number} 行 metadata 是 dict，"
                f"实际为 {type(row_data)}"
            )
        if "text" not in row_data:
            raise ValueError(f"第 {row_number} 行 metadata 缺失 text")
        raw_text = row_data["text"]
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ValueError(
                f"第 {row_number} 行 metadata.text 必须是非空字符串"
            )
        tokens = tokenizer(raw_text)
        if not isinstance(tokens, list) or not tokens:
            raise ValueError(
                f"第 {row_number} 行 metadata.text 未产生有效 token"
            )
        tokenized_corpus.append(tokens)

    return BM25Okapi(
        tokenized_corpus,
        k1=float(k1),
        b=float(b),
        epsilon=float(epsilon),
    )


def sparse_retrieve(
    query,
    bm25_index,
    metadata,
    *,
    top_k=5,
    tokenizer=tokenize_text,
):
    """执行 BM25 检索，返回与 Dense Retrieval 兼容的结果。"""
    if not isinstance(query, str):
        raise TypeError("query 必须是字符串")
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query 不能为空或只包含空白字符")
    if not isinstance(metadata, list):
        raise TypeError("metadata 必须是列表")
    if not metadata:
        raise ValueError("metadata 不能为空")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k 必须是正整数")
    if not callable(tokenizer):
        raise TypeError("tokenizer 必须是可调用对象")

    get_scores = getattr(bm25_index, "get_scores", None)
    if not callable(get_scores):
        raise TypeError("bm25_index 必须提供可调用的 get_scores()")

    query_tokens = tokenizer(normalized_query)
    scores = get_scores(query_tokens)
    try:
        score_count = len(scores)
    except TypeError as exc:
        raise ValueError("BM25 get_scores() 必须返回分数序列") from exc
    if score_count != len(metadata):
        raise ValueError(
            "BM25 分数数量必须与 metadata 数量一致："
            f"scores={score_count}, metadata={len(metadata)}"
        )

    ranked_positions = sorted(
        (
            (position, float(score))
            for position, score in enumerate(scores)
            if float(score) > 0
        ),
        key=lambda item: (-item[1], item[0]),
    )[: min(top_k, len(metadata))]

    results = []
    for rank, (position, score) in enumerate(ranked_positions, start=1):
        result = copy.deepcopy(metadata[position])
        result.update(
            {
                "index_id": int(position),
                "rank": rank,
                "score": score,
                "retrieval_source": "bm25",
            }
        )
        results.append(result)
    return results
