from src.rag_qa import answer_question
import src.retriever as retriever


def test_neighbor_expansion_flows_into_answer_context(monkeypatch):
    metadata = [
        {
            "doc_id": "report-1",
            "chunk_id": f"report-1:p1:c{index}",
            "document_chunk_index": index,
            "page_number": 1,
            "text": text,
        }
        for index, text in enumerate(
            ["上文背景", "核心证据", "下文解释"]
        )
    ]

    monkeypatch.setattr(
        retriever,
        "search_index",
        lambda **_kwargs: [
            {
                **metadata[1],
                "rank": 1,
                "score": 1.0,
            }
        ],
    )
    retrieval_results = retriever.retrieve(
        query="测试问题",
        index=object(),
        metadata=metadata,
        embed_batch=lambda _texts: [[1.0, 0.0]],
        top_k=3,
        neighbor_window=1,
        neighbor_seed_top_n=1,
        neighbor_decay=0.8,
    )

    captured = {}

    def fake_generate_answer(*, system_prompt, user_prompt):
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        return " 基于连续上下文的回答 "

    answer = answer_question(
        question="测试问题",
        retrieval_results=retrieval_results,
        generate_answer=fake_generate_answer,
    )

    assert answer == "基于连续上下文的回答"
    assert "核心证据" in captured["user_prompt"]
    assert "上文背景" in captured["user_prompt"]
    assert "下文解释" in captured["user_prompt"]
    assert sum(
        item["is_direct_hit"] is False for item in retrieval_results
    ) == 2
