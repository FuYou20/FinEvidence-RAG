import pytest

from src.evidence_protocol import EvidenceBundle, EvidenceRequest
from src.stance_research import handle_stance_research


def make_request(*, direction="support", research_focus=None):
    return EvidenceRequest(
        request_id="req-stance",
        requester="supporting_agent",
        mode="stance_research",
        topic="传音是否具有长期投资价值",
        direction=direction,
        research_focus=(
            ["竞争壁垒", "盈利能力"]
            if research_focus is None
            else research_focus
        ),
        max_deepening_steps=2,
    )


def make_investigation_result(*, status="sufficient"):
    if status == "sufficient":
        return {
            "status": "sufficient",
            "question": "调查问题",
            "final_query": "最终查询",
            "evidence": [
                {
                    "evidence_id": "ev-1",
                    "text": "传音具有本土化渠道优势。",
                }
            ],
            "steps": [
                {
                    "step_number": 1,
                    "action": "rewrite_query",
                    "reason": "初始检索不足",
                    "query_before": "原始查询",
                    "query_after": "本土化 渠道",
                    "added_evidence_count": 1,
                }
            ],
            "stop_reason": "evidence_sufficient",
            "last_decision": {
                "enough": True,
                "next_action": "finish",
                "reason": "证据已经充分",
            },
        }

    return {
        "status": "insufficient",
        "question": "调查问题",
        "final_query": "最终查询",
        "evidence": [],
        "steps": [],
        "stop_reason": "no_progress",
        "last_decision": {
            "enough": False,
            "next_action": "rewrite_query",
            "reason": "没有找到与盈利能力相关的数据",
            "next_query": "盈利能力 数据",
        },
    }


def test_support_request_calls_runner_and_returns_sufficient_bundle():
    captured = {}

    def investigation_runner(**kwargs):
        captured.update(kwargs)
        return make_investigation_result()

    bundle = handle_stance_research(
        make_request(),
        investigation_runner,
    )

    assert isinstance(bundle, EvidenceBundle)
    assert bundle.request_id == "req-stance"
    assert bundle.mode == "stance_research"
    assert bundle.status == "sufficient"
    assert bundle.evidence[0]["evidence_id"] == "ev-1"
    assert captured["max_deepening_steps"] == 2
    assert "寻找支持该议题" in captured["question"]
    assert "竞争壁垒、盈利能力" in captured["question"]
    assert "不得隐瞒" in captured["question"]


def test_oppose_request_builds_opposing_investigation_question():
    captured = {}

    def investigation_runner(**kwargs):
        captured.update(kwargs)
        return make_investigation_result()

    handle_stance_research(
        make_request(direction="oppose", research_focus=[]),
        investigation_runner,
    )

    assert "寻找反对或质疑该议题" in captured["question"]
    assert "无指定重点" in captured["question"]


def test_insufficient_result_maps_reason_to_missing_information():
    bundle = handle_stance_research(
        make_request(),
        lambda **_kwargs: make_investigation_result(
            status="insufficient"
        ),
    )

    assert bundle.status == "insufficient"
    assert bundle.evidence == []
    assert bundle.missing_information == [
        "没有找到与盈利能力相关的数据"
    ]


def test_bundle_contains_investigation_trace():
    bundle = handle_stance_research(
        make_request(),
        lambda **_kwargs: make_investigation_result(),
    )

    assert bundle.investigation_trace[0]["action"] == "rewrite_query"
    assert bundle.investigation_trace[-1]["stop_reason"] == (
        "evidence_sufficient"
    )
    assert bundle.investigation_trace[-1]["final_query"] == "最终查询"


def test_accepts_request_dictionary():
    bundle = handle_stance_research(
        make_request().model_dump(),
        lambda **_kwargs: make_investigation_result(),
    )

    assert bundle.request_id == "req-stance"


def test_context_evidence_is_resolved_as_seed_for_fresh_research():
    request = make_request().model_copy(
        update={"context_evidence_ids": ["E-original"]}
    )
    captured = {}

    def investigation_runner(**kwargs):
        captured.update(kwargs)
        return make_investigation_result()

    handle_stance_research(
        request,
        investigation_runner,
        resolve_evidence=lambda ids: [
            {"evidence_id": ids[0], "text": "原观点证据"}
        ],
    )

    assert captured["seed_evidence"][0]["evidence_id"] == "E-original"


def test_rejects_verify_evidence_request():
    request = {
        "request_id": "req-verify",
        "requester": "opposing_agent",
        "mode": "verify_evidence",
        "evidence_ids": ["S1"],
        "verification_goal": "检查来源与时效性",
    }

    with pytest.raises(ValueError, match="stance_research"):
        handle_stance_research(
            request,
            lambda **_kwargs: make_investigation_result(),
        )


def test_rejects_non_callable_runner():
    with pytest.raises(TypeError, match="investigation_runner"):
        handle_stance_research(make_request(), None)


def test_propagates_runner_exception_unchanged():
    runner_error = RuntimeError("调查工作流暂时不可用")

    def failing_runner(**_kwargs):
        raise runner_error

    with pytest.raises(RuntimeError) as exc_info:
        handle_stance_research(make_request(), failing_runner)

    assert exc_info.value is runner_error
