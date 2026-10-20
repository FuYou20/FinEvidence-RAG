from types import SimpleNamespace

import pytest

from errors import InvalidCitationError
from src.evaluation.rag_benchmark_generation import (
    BenchmarkAnswer,
    create_benchmark_answer_generator,
    generate_prediction_record,
)


def retrieval_results():
    return [
        {
            "doc_id": "doc_a",
            "page_number": 1,
            "chunk_id": "doc_a:p1:c0",
            "content_type": "text",
            "text": "甲公司收入为10亿元。",
        },
        {
            "doc_id": "doc_b",
            "page_number": 2,
            "chunk_id": "doc_b:p2:c0",
            "content_type": "text",
            "text": "乙公司收入为20亿元。",
        },
    ]


def test_generate_prediction_maps_temporary_sources_to_stable_ids():
    case = {
        "query_id": "enum_1",
        "question": "分别列出收入",
        "task_type": "enumeration",
        "reference_answer": "甲10亿元，乙20亿元",
    }

    def fake_generate(**_kwargs):
        return {
            "predicted_action": "answer_with_citations",
            "answer_text": "甲10亿元，乙20亿元。",
            "cited_source_ids": ["S1", "S2"],
            "predicted_items": [
                {"item": "甲10亿元", "source_ids": ["S1"]},
                {"item": "乙20亿元", "source_ids": ["S2"]},
            ],
        }

    record = generate_prediction_record(
        case=case,
        retrieval_results=retrieval_results(),
        generate_answer=fake_generate,
    )

    assert record["cited_doc_ids"] == ["doc_a", "doc_b"]
    assert record["cited_chunk_ids"] == ["doc_a:p1:c0", "doc_b:p2:c0"]
    assert record["predicted_items"][1]["cited_chunk_ids"] == ["doc_b:p2:c0"]


def test_generate_prediction_rejects_unknown_source():
    def fake_generate(**_kwargs):
        return {
            "predicted_action": "answer_with_citations",
            "answer_text": "答案",
            "cited_source_ids": ["S9"],
            "predicted_items": [],
        }

    with pytest.raises(InvalidCitationError, match="S9"):
        generate_prediction_record(
            case={
                "query_id": "fact_1",
                "question": "问题",
                "task_type": "single_fact",
                "reference_answer": "答案",
            },
            retrieval_results=retrieval_results(),
            generate_answer=fake_generate,
        )


def test_answer_generator_uses_structured_parse():
    captured = {}

    class FakeResponses:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                output_parsed=BenchmarkAnswer(
                    predicted_action="abstain",
                    answer_text="证据不足。",
                    cited_source_ids=[],
                    predicted_items=[],
                )
            )

    generator = create_benchmark_answer_generator(
        SimpleNamespace(responses=FakeResponses()),
        model_name="test-model",
    )
    answer = generator(question="问题", evidence_context="[S1] 证据")

    assert answer.predicted_action == "abstain"
    assert captured["model"] == "test-model"
    assert captured["text_format"] is BenchmarkAnswer
