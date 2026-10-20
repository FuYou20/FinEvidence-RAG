from .evidence_protocol import EvidenceBundle, EvidenceRequest


def dispatch_evidence_request(
    request,
    stance_research,
    verify_evidence,
):
    """根据请求模式调用对应调查处理器，并验证返回协议。

    Args:
        request: ``EvidenceRequest`` 对象或可由其校验的字典。
        stance_research: 处理立场证据调查的可调用对象。
        verify_evidence: 处理证据查证的可调用对象。

    Returns:
        与请求 request_id、mode 一致的 ``EvidenceBundle``。

    Raises:
        TypeError: 处理器不是可调用对象。
        ValidationError: request 或处理器返回值不符合 Pydantic 协议。
        ValueError: 返回结果的 request_id 或 mode 与请求不一致。
    """
    validated_request = EvidenceRequest.model_validate(request)
    if not callable(stance_research):
        raise TypeError("stance_research 必须是可调用对象")
    if not callable(verify_evidence):
        raise TypeError("verify_evidence 必须是可调用对象")

    if validated_request.mode == "stance_research":
        raw_bundle = stance_research(
            request=validated_request,
        )
    elif validated_request.mode == "verify_evidence":
        raw_bundle = verify_evidence(
            request=validated_request,
        )
    validated_bundle = EvidenceBundle.model_validate(raw_bundle)
    if validated_bundle.request_id != validated_request.request_id:
        raise ValueError("返回结果的 request_id 与请求不一致")
    if validated_request.mode != validated_bundle.mode:
        raise ValueError("返回结果的 mode 与请求不一致")

    return validated_bundle
