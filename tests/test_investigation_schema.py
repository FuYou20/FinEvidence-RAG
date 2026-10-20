import pytest
from pydantic import ValidationError

from src.investigation_schema import (
    EvidenceDecision,
    InvestigationResult,
)


def test_evidence_decision_normalizes_reason():
    decision = EvidenceDecision(
        enough=True,
        next_action="finish",
        reason="  证据已经充分  ",
    )

    assert decision.reason == "证据已经充分"


@pytest.mark.parametrize(
    "payload",
    [
        {
            "enough": True,
            "next_action": "web_search",
            "reason": "不一致",
        },
        {
            "enough": False,
            "next_action": "finish",
            "reason": "尚不充分",
        },
        {
            "enough": False,
            "next_action": "rewrite_query",
            "reason": "   ",
        },
    ],
)
def test_evidence_decision_rejects_invalid_combinations(payload):
    with pytest.raises(ValidationError):
        EvidenceDecision.model_validate(payload)


def test_investigation_result_rejects_inconsistent_stop_reason():
    with pytest.raises(ValidationError):
        InvestigationResult(
            status="sufficient",
            question="问题",
            final_query="问题",
            evidence=[],
            steps=[],
            stop_reason="no_progress",
            last_decision={
                "enough": True,
                "next_action": "finish",
                "reason": "已经充分",
            },
        )
