import pytest

from src.evidence_protocol import EvidenceBundle, EvidenceRequest
from src.evidence_verification import handle_verify_evidence


def make_request():
    return EvidenceRequest(
        request_id="req-verify",
        requester="opposing_agent",
        mode="verify_evidence",
        evidence_ids=["old-S3"],
        verification_goal=(
            "检查原始出处、时效性和统计口径，并寻找独立来源"
        ),
        max_deepening_steps=2,
    )


def make_investigation_result(*, status="sufficient"):
    evidence = [
        {
            "evidence_id": "old-S3",
            "text": "2022年研报中的原始材料。",
            "published_at": "2022-04-22",
        },
        {
            "evidence_id": "new-S1",
            "text": "独立来源提供的更新材料。",
            "published_at": "2026-08-01",
        },
    ]
    if status == "sufficient":
        return {
            "status": "sufficient",
            "question": "证据查证问题",
            "final_query": "证据S3 原始来源 最新同口径数据",
            "evidence": evidence,
            "steps": [],
            "stop_reason": "evidence_sufficient",
            "last_decision": {
                "enough": True,
                "next_action": "finish",
                "reason": "已定位原文并找到独立来源",
            },
        }

    return {
        "status": "insufficient",
        "question": "证据查证问题",
        "final_query": "证据S3 原始来源",
        "evidence": evidence[:1],
        "steps": [],
        "stop_reason": "no_progress",
        "last_decision": {
            "enough": False,
            "next_action": "web_search",
            "reason": "尚未找到最新的独立来源",
            "next_query": "最新 独立来源",
        },
    }


def test_returns_neutral_sufficient_evidence_bundle():
    bundle = handle_verify_evidence(
        make_request(),
        lambda **_kwargs: make_investigation_result(),
    )

    assert isinstance(bundle, EvidenceBundle)
    assert bundle.mode == "verify_evidence"
    assert bundle.status == "sufficient"
    assert [item["evidence_id"] for item in bundle.evidence] == [
        "old-S3",
        "new-S1",
    ]
    assert bundle.missing_information == []


def test_insufficient_result_keeps_partial_evidence_and_gap():
    bundle = handle_verify_evidence(
        make_request(),
        lambda **_kwargs: make_investigation_result(
            status="insufficient"
        ),
    )

    assert bundle.status == "insufficient"
    assert bundle.evidence[0]["evidence_id"] == "old-S3"
    assert bundle.missing_information == [
        "尚未找到最新的独立来源"
    ]


def test_runner_receives_neutral_verification_question_and_step_limit():
    captured = {}

    def investigation_runner(**kwargs):
        captured.update(kwargs)
        return make_investigation_result()

    handle_verify_evidence(make_request(), investigation_runner)

    assert captured["max_deepening_steps"] == 2
    assert "old-S3" in captured["question"]
    assert make_request().verification_goal in captured["question"]
    assert "原始来源" in captured["question"]
    assert "独立来源" in captured["question"]
    assert "不判断任何投资观点" in captured["question"]


def test_rejects_non_verify_evidence_request():
    request = {
        "request_id": "req-stance",
        "requester": "supporting_agent",
        "mode": "stance_research",
        "topic": "辩论议题",
        "direction": "support",
    }

    with pytest.raises(ValueError, match="verify_evidence"):
        handle_verify_evidence(
            request,
            lambda **_kwargs: make_investigation_result(),
        )


def test_rejects_non_callable_investigation_runner():
    with pytest.raises(TypeError, match="investigation_runner"):
        handle_verify_evidence(make_request(), None)
