import pytest
from pydantic import ValidationError

from src.evidence_protocol import EvidenceBundle, EvidenceRequest


def test_stance_request_normalizes_and_deduplicates_fields():
    request = EvidenceRequest(
        request_id="  req-001  ",
        requester="  supporting_agent  ",
        mode="stance_research",
        topic="  传音是否具有长期投资价值  ",
        direction="support",
        research_focus=[" 竞争壁垒 ", "竞争壁垒", " 盈利能力 "],
        context_evidence_ids=[" E-old ", "E-old"],
    )

    assert request.request_id == "req-001"
    assert request.requester == "supporting_agent"
    assert request.topic == "传音是否具有长期投资价值"
    assert request.research_focus == ["竞争壁垒", "盈利能力"]
    assert request.context_evidence_ids == ["E-old"]


def test_verify_evidence_request_normalizes_fields():
    request = EvidenceRequest(
        request_id="req-002",
        requester="opposing_agent",
        mode="verify_evidence",
        evidence_ids=[" S3 ", "S3"],
        verification_goal="  检查来源与时效性  ",
    )

    assert request.evidence_ids == ["S3"]
    assert request.verification_goal == "检查来源与时效性"


@pytest.mark.parametrize(
    "payload, expected_message",
    [
        (
            {"mode": "stance_research", "direction": "support"},
            "topic",
        ),
        (
            {"mode": "stance_research", "topic": "辩论议题"},
            "direction",
        ),
        (
            {
                "mode": "stance_research",
                "topic": "辩论议题",
                "direction": "oppose",
                "evidence_ids": ["S1"],
            },
            "evidence_ids",
        ),
        (
            {
                "mode": "stance_research",
                "topic": "辩论议题",
                "direction": "oppose",
                "verification_goal": "不应出现",
            },
            "verification_goal",
        ),
        (
            {
                "mode": "verify_evidence",
                "verification_goal": "检查来源",
            },
            "evidence_ids",
        ),
        (
            {
                "mode": "verify_evidence",
                "evidence_ids": ["S1"],
            },
            "verification_goal",
        ),
        (
            {
                "mode": "verify_evidence",
                "evidence_ids": ["S1"],
                "verification_goal": "检查来源",
                "direction": "support",
            },
            "direction",
        ),
        (
            {
                "mode": "verify_evidence",
                "evidence_ids": ["S1"],
                "verification_goal": "检查来源",
                "context_evidence_ids": ["S2"],
            },
            "context_evidence_ids",
        ),
    ],
)
def test_request_rejects_fields_that_do_not_match_mode(
    payload,
    expected_message,
):
    with pytest.raises(ValidationError, match=expected_message):
        EvidenceRequest.model_validate(
            {
                "request_id": "req-invalid",
                "requester": "test_agent",
                **payload,
            }
        )


def test_request_lists_do_not_share_mutable_defaults():
    first = EvidenceRequest(
        request_id="req-1",
        requester="judge_agent",
        mode="verify_evidence",
        evidence_ids=["S1"],
        verification_goal="检查来源",
    )
    second = EvidenceRequest(
        request_id="req-2",
        requester="judge_agent",
        mode="verify_evidence",
        evidence_ids=["S2"],
        verification_goal="检查时效",
    )

    first.evidence_ids.append("S3")

    assert second.evidence_ids == ["S2"]


@pytest.mark.parametrize(
    "mode",
    ["stance_research", "verify_evidence"],
)
def test_bundle_accepts_sufficient_evidence_for_both_modes(mode):
    bundle = EvidenceBundle(
        request_id="req-1",
        mode=mode,
        status="sufficient",
        evidence=[{"evidence_id": "ev-1", "text": "证据"}],
    )

    assert bundle.status == "sufficient"


def test_insufficient_bundle_preserves_partial_evidence_and_gap():
    bundle = EvidenceBundle(
        request_id="req-2",
        mode="verify_evidence",
        status="insufficient",
        evidence=[{"evidence_id": "ev-1", "text": "部分证据"}],
        missing_information=["缺少独立来源验证"],
    )

    assert bundle.evidence[0]["evidence_id"] == "ev-1"
    assert bundle.missing_information == ["缺少独立来源验证"]


@pytest.mark.parametrize(
    "payload",
    [
        {
            "mode": "stance_research",
            "status": "sufficient",
            "evidence": [],
        },
        {
            "mode": "verify_evidence",
            "status": "insufficient",
            "missing_information": [],
        },
        {
            "mode": "verify_evidence",
            "status": "supported",
            "evidence": [{"text": "旧观点结论不应再出现"}],
        },
    ],
)
def test_bundle_rejects_inconsistent_or_opinionated_results(payload):
    with pytest.raises(ValidationError):
        EvidenceBundle.model_validate(
            {
                "request_id": "req-invalid",
                **payload,
            }
        )
