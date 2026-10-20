import pytest
from pydantic import ValidationError

from src.evidence_dispatcher import dispatch_evidence_request
from src.evidence_protocol import EvidenceBundle, EvidenceRequest


def make_stance_request():
    return EvidenceRequest(
        request_id="req-stance",
        requester="supporting_agent",
        mode="stance_research",
        topic="传音是否具有长期投资价值",
        direction="support",
        research_focus=["竞争壁垒"],
    )


def make_verify_request():
    return EvidenceRequest(
        request_id="req-verify",
        requester="opposing_agent",
        mode="verify_evidence",
        evidence_ids=["S3"],
        verification_goal="检查来源和时效性",
    )


def make_stance_bundle(request_id="req-stance"):
    return {
        "request_id": request_id,
        "mode": "stance_research",
        "status": "sufficient",
        "evidence": [{"evidence_id": "ev-1", "text": "相关证据"}],
    }


def make_verify_bundle(request_id="req-verify"):
    return {
        "request_id": request_id,
        "mode": "verify_evidence",
        "status": "sufficient",
        "evidence": [
            {"evidence_id": "ev-2", "text": "查证所得材料"}
        ],
    }


def test_dispatches_stance_request_to_stance_handler_only():
    calls = {"stance": 0, "verify": 0}
    request = make_stance_request()

    def stance_research(*, request):
        calls["stance"] += 1
        assert isinstance(request, EvidenceRequest)
        assert request.mode == "stance_research"
        return make_stance_bundle()

    def verify_evidence(**_kwargs):
        calls["verify"] += 1
        return make_verify_bundle()

    result = dispatch_evidence_request(
        request,
        stance_research,
        verify_evidence,
    )

    assert isinstance(result, EvidenceBundle)
    assert result.status == "sufficient"
    assert calls == {"stance": 1, "verify": 0}


def test_dispatches_verify_request_to_verify_handler_only():
    calls = {"stance": 0, "verify": 0}

    def stance_research(**_kwargs):
        calls["stance"] += 1
        return make_stance_bundle()

    def verify_evidence(*, request):
        calls["verify"] += 1
        assert request.evidence_ids == ["S3"]
        return make_verify_bundle()

    result = dispatch_evidence_request(
        make_verify_request().model_dump(),
        stance_research,
        verify_evidence,
    )

    assert result.status == "sufficient"
    assert calls == {"stance": 0, "verify": 1}


@pytest.mark.parametrize("handler_name", ["stance", "verify"])
def test_rejects_non_callable_handler(handler_name):
    stance_handler = (
        None if handler_name == "stance" else lambda **_kwargs: {}
    )
    verify_handler = (
        None if handler_name == "verify" else lambda **_kwargs: {}
    )

    with pytest.raises(TypeError, match=f"{handler_name}_"):
        dispatch_evidence_request(
            make_stance_request(),
            stance_handler,
            verify_handler,
        )


def test_rejects_bundle_with_mismatched_request_id():
    with pytest.raises(ValueError, match="request_id"):
        dispatch_evidence_request(
            make_stance_request(),
            lambda **_kwargs: make_stance_bundle("another-request"),
            lambda **_kwargs: make_verify_bundle(),
        )


def test_rejects_bundle_with_mismatched_mode():
    invalid_bundle = make_verify_bundle(request_id="req-stance")

    with pytest.raises(ValueError, match="mode"):
        dispatch_evidence_request(
            make_stance_request(),
            lambda **_kwargs: invalid_bundle,
            lambda **_kwargs: make_verify_bundle(),
        )


def test_rejects_handler_result_that_breaks_bundle_protocol():
    with pytest.raises(ValidationError):
        dispatch_evidence_request(
            make_verify_request(),
            lambda **_kwargs: make_stance_bundle(),
            lambda **_kwargs: {
                "request_id": "req-verify",
                "mode": "verify_evidence",
                "status": "sufficient",
                "evidence": [],
            },
        )


def test_propagates_handler_exception_unchanged():
    handler_error = RuntimeError("调查服务暂时不可用")

    def failing_handler(**_kwargs):
        raise handler_error

    with pytest.raises(RuntimeError) as exc_info:
        dispatch_evidence_request(
            make_stance_request(),
            failing_handler,
            lambda **_kwargs: make_verify_bundle(),
        )

    assert exc_info.value is handler_error
