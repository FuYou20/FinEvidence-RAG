from .evidence_protocol import EvidenceBundle, EvidenceRequest
from .investigation_schema import InvestigationResult


def build_stance_investigation_question(request):
    """将立场调查请求转换为调查工作流使用的问题文本。"""
    direction_text = {
        "support": "寻找支持该议题的事实和数据",
        "oppose": "寻找反对或质疑该议题的事实和数据",
    }[request.direction]
    focus_text = (
        "、".join(request.research_focus)
        if request.research_focus
        else "无指定重点"
    )
    return (
        f"议题：{request.topic}\n"
        f"调查方向：{direction_text}\n"
        f"调查重点：{focus_text}\n"
        "要求：优先寻找与调查方向相关的可引用证据，"
        "但不得隐瞒检索过程中发现的相反或冲突证据。"
    )


def _build_investigation_trace(result):
    trace = [step.model_dump() for step in result.steps]
    trace.append(
        {
            "stop_reason": result.stop_reason,
            "final_query": result.final_query,
            "last_decision": result.last_decision.model_dump(),
        }
    )
    return trace


def handle_stance_research(
    request,
    investigation_runner,
    *,
    resolve_evidence=None,
):
    """执行立场证据调查，并转换为统一的 ``EvidenceBundle``。

    ``investigation_runner`` 是已经绑定检索、判断、Query Rewrite、
    多模态和联网工具的调查工作流入口。它接受 ``question`` 与
    ``max_deepening_steps`` 两个关键字参数，并返回
    ``InvestigationResult`` 或符合该协议的字典。
    """
    validated_request = EvidenceRequest.model_validate(request)
    if validated_request.mode != "stance_research":
        raise ValueError(
            "handle_stance_research 只接受 stance_research 请求"
        )
    if not callable(investigation_runner):
        raise TypeError("investigation_runner 必须是可调用对象")
    if resolve_evidence is not None and not callable(resolve_evidence):
        raise TypeError("resolve_evidence 必须是可调用对象或 None")

    question = build_stance_investigation_question(validated_request)
    runner_kwargs = {
        "question": question,
        "max_deepening_steps": validated_request.max_deepening_steps,
    }
    if validated_request.context_evidence_ids:
        if resolve_evidence is None:
            raise ValueError("存在 context_evidence_ids 时必须提供 resolve_evidence")
        runner_kwargs["seed_evidence"] = resolve_evidence(
            validated_request.context_evidence_ids
        )
    raw_result = investigation_runner(
        **runner_kwargs,
    )
    result = InvestigationResult.model_validate(raw_result)
    status = result.status
    missing_information = []
    if status == "insufficient":
        missing_information = [result.last_decision.reason]

    return EvidenceBundle(
        request_id=validated_request.request_id,
        mode=validated_request.mode,
        status=status,
        evidence=result.evidence,
        missing_information=missing_information,
        investigation_trace=_build_investigation_trace(result),
    )
