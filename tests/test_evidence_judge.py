import pytest

from src.evidence_judge import evaluate_evidence
from src.investigation_schema import EvidenceDecision


def make_evidence():
    return [
        {
            "doc_id": "report_001",
            "page_number": 15,
            "chunk_id": "report_001:p15:c0",
            "content_type": "text",
            "text": "具体市场份额见图表27。",
        }
    ]


def test_evaluate_evidence_builds_prompt_and_returns_decision():
    captured = {}

    def fake_generate_decision(*, system_prompt, user_prompt):
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        return {
            "enough": False,
            "next_action": "multimodal",
            "reason": "关键数值位于图表中",
        }

    decision = evaluate_evidence(
        question="目前的市场份额是多少？",
        query="传音 非洲 市场份额",
        evidence=make_evidence(),
        generate_decision=fake_generate_decision,
    )

    assert isinstance(decision, EvidenceDecision)
    assert decision.next_action == "multimodal"
    assert "[S1]" in captured["user_prompt"]
    assert "当前检索 Query" in captured["user_prompt"]
    assert "一次只能选择一个动作" in captured["system_prompt"]


def test_evaluate_evidence_supports_empty_evidence():
    captured = {}

    def fake_generate_decision(**kwargs):
        captured.update(kwargs)
        return {
            "enough": False,
            "next_action": "rewrite_query",
            "reason": "没有检索到证据",
            "next_query": "传音 非洲 渠道优势",
        }

    decision = evaluate_evidence(
        "传音有哪些竞争优势？",
        [],
        fake_generate_decision,
    )

    assert decision.next_action == "rewrite_query"
    assert "没有检索到任何证据" in captured["user_prompt"]


def test_evaluate_evidence_bounds_accumulated_evidence_prompt():
    captured = {}
    evidence = [
        {
            "evidence_id": f"E{index}",
            "doc_id": "report_001",
            "page_number": index,
            "chunk_id": f"c{index}",
            "text": str(index) * 300,
        }
        for index in range(1, 5)
    ]

    def fake_generate_decision(**kwargs):
        captured.update(kwargs)
        return {
            "enough": True,
            "next_action": "finish",
            "reason": "已有足够证据",
        }

    evaluate_evidence(
        "问题",
        evidence,
        fake_generate_decision,
        max_evidence_items=2,
        max_evidence_text_chars=120,
        max_chars_per_evidence=80,
    )

    prompt = captured["user_prompt"]
    assert "evidence_id：E1" in prompt
    assert "evidence_id：E2" in prompt
    assert "evidence_id：E3" not in prompt
    assert sum(prompt.count(str(index) * 50) for index in range(1, 5)) <= 2


@pytest.mark.parametrize("question", [None, 123, "   "])
def test_evaluate_evidence_rejects_invalid_question(question):
    expected_error = TypeError if not isinstance(question, str) else ValueError
    with pytest.raises(expected_error, match="question"):
        evaluate_evidence(
            question,
            [],
            lambda **_kwargs: {},
        )


def test_evaluate_evidence_rejects_non_callable_generator():
    with pytest.raises(TypeError, match="generate_decision"):
        evaluate_evidence("问题", [], None)
