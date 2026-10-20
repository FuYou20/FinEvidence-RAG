from copy import deepcopy

from .evidence_protocol import EvidenceBundle, EvidenceRequest
from .investigation_schema import InvestigationResult
from .stance_research import _build_investigation_trace


def build_evidence_verification_question(request):
    """将证据查证请求转换为中立调查问题。"""
    target_text = "、".join(request.evidence_ids)
    return (
        f"待查证证据：{target_text}\n"
        f"查证目标：{request.verification_goal}\n"
        "要求：定位原始来源并核对引用内容；检查发布时间、对象、"
        "指标、数值、单位与统计口径；尽可能寻找独立来源交叉验证；"
        "如来源之间存在不一致，完整保留各方材料；只返回可追溯证据"
        "和未解决信息，不判断任何投资观点是否正确。"
    )


def handle_verify_evidence(
    request,
    investigation_runner,
    *,
    resolve_evidence=None,
):
    """查证指定证据的来源、准确性、时效性和独立验证情况。"""
    validated_request = EvidenceRequest.model_validate(request)
    if validated_request.mode != "verify_evidence":
        raise ValueError(
            "handle_verify_evidence 只接受 verify_evidence 请求"
        )
    if not callable(investigation_runner):
        raise TypeError("investigation_runner 必须是可调用对象")
    if resolve_evidence is not None and not callable(resolve_evidence):
        raise TypeError("resolve_evidence 必须是可调用对象或 None")

    question = build_evidence_verification_question(validated_request)
    runner_kwargs = {
        "question": question,
        "max_deepening_steps": validated_request.max_deepening_steps,
    }
    if resolve_evidence is not None:
        runner_kwargs["initial_evidence"] = resolve_evidence(
            validated_request.evidence_ids
        )
    raw_result = investigation_runner(
        **runner_kwargs,
    )
    result = InvestigationResult.model_validate(raw_result)

    missing_information = []
    if result.status == "insufficient":
        missing_information = [result.last_decision.reason]

    return EvidenceBundle(
        request_id=validated_request.request_id,
        mode=validated_request.mode,
        status=result.status,
        evidence=deepcopy(result.evidence),
        missing_information=missing_information,
        investigation_trace=_build_investigation_trace(result),
    )
