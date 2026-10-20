from copy import deepcopy

from errors import RequestError

from .investigation_schema import (
    EvidenceDecision,
    InvestigationResult,
    InvestigationStep,
)


def _normalize_required_string(value, *, field_name):
    if not isinstance(value, str):
        raise TypeError(f"{field_name} 必须是字符串")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} 不能为空")
    return normalized


def _normalize_evidence_batch(value, *, source_name):
    if not isinstance(value, list):
        raise TypeError(f"{source_name} 必须返回 list")

    normalized = []
    for position, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise TypeError(
                f"{source_name} 返回的第 {position} 条证据必须是 dict"
            )
        normalized.append(deepcopy(item))
    return normalized


def _merge_evidence(existing, additions):
    merged = deepcopy(existing)
    added_count = 0
    for item in additions:
        if item not in merged:
            merged.append(deepcopy(item))
            added_count += 1
    return merged, added_count


def _build_result(
    *,
    status,
    question,
    final_query,
    evidence,
    steps,
    stop_reason,
    last_decision,
):
    return InvestigationResult(
        status=status,
        question=question,
        final_query=final_query,
        evidence=deepcopy(evidence),
        steps=deepcopy(steps),
        stop_reason=stop_reason,
        last_decision=last_decision,
    )


def run_evidence_investigation(
    question,
    retrieve_evidence,
    judge_evidence,
    *,
    initial_evidence=None,
    seed_evidence=None,
    rewrite_query=None,
    parse_multimodal=None,
    search_web=None,
    max_deepening_steps=2,
):
    """运行“一级检索 + 按需深入调查”的受限证据工作流。

    所有外部能力都以可调用对象注入。初始阶段始终执行文本/表格检索；
    此后判断器每轮只能选择 rewrite_query、multimodal、web_search
    中的一项。工作流在证据充分、没有新证据、动作不可用或达到步数
    上限时停止。
    """
    normalized_question = _normalize_required_string(
        question,
        field_name="question",
    )
    if not callable(retrieve_evidence):
        raise TypeError("retrieve_evidence 必须是可调用对象")
    if not callable(judge_evidence):
        raise TypeError("judge_evidence 必须是可调用对象")
    for name, action in (
        ("rewrite_query", rewrite_query),
        ("parse_multimodal", parse_multimodal),
        ("search_web", search_web),
    ):
        if action is not None and not callable(action):
            raise TypeError(f"{name} 必须是可调用对象或 None")
    if (
        not isinstance(max_deepening_steps, int)
        or isinstance(max_deepening_steps, bool)
        or max_deepening_steps < 0
    ):
        raise ValueError("max_deepening_steps 必须是非负整数")

    current_query = normalized_question
    if initial_evidence is not None and seed_evidence is not None:
        raise ValueError("initial_evidence 与 seed_evidence 不能同时提供")

    if initial_evidence is None:
        retrieved = _normalize_evidence_batch(
            retrieve_evidence(query=current_query),
            source_name="retrieve_evidence",
        )
        if seed_evidence is None:
            evidence = retrieved
        else:
            seeds = _normalize_evidence_batch(
                seed_evidence,
                source_name="seed_evidence",
            )
            evidence, _ = _merge_evidence(seeds, retrieved)
    else:
        evidence = _normalize_evidence_batch(
            initial_evidence,
            source_name="initial_evidence",
        )
    steps = []

    while True:
        raw_decision = judge_evidence(
            question=normalized_question,
            query=current_query,
            evidence=deepcopy(evidence),
        )
        decision = EvidenceDecision.model_validate(raw_decision)

        if decision.enough:
            return _build_result(
                status="sufficient",
                question=normalized_question,
                final_query=current_query,
                evidence=evidence,
                steps=steps,
                stop_reason="evidence_sufficient",
                last_decision=decision,
            )

        if len(steps) >= max_deepening_steps:
            return _build_result(
                status="insufficient",
                question=normalized_question,
                final_query=current_query,
                evidence=evidence,
                steps=steps,
                stop_reason="max_deepening_steps",
                last_decision=decision,
            )

        action = decision.next_action
        query_before = current_query

        if action == "rewrite_query":
            current_query = decision.next_query
            additions = _normalize_evidence_batch(
                retrieve_evidence(query=current_query),
                source_name="retrieve_evidence",
            )
        elif action == "multimodal":
            if parse_multimodal is None:
                return _build_result(
                    status="insufficient",
                    question=normalized_question,
                    final_query=current_query,
                    evidence=evidence,
                    steps=steps,
                    stop_reason="action_unavailable",
                    last_decision=decision,
                )
            additions = _normalize_evidence_batch(
                parse_multimodal(
                    question=normalized_question,
                    query=current_query,
                    evidence=deepcopy(evidence),
                ),
                source_name="parse_multimodal",
            )
        elif action == "web_search":
            if search_web is None:
                return _build_result(
                    status="insufficient",
                    question=normalized_question,
                    final_query=current_query,
                    evidence=evidence,
                    steps=steps,
                    stop_reason="action_unavailable",
                    last_decision=decision,
                )
            try:
                current_query = decision.next_query
                additions = _normalize_evidence_batch(
                    search_web(
                        question=normalized_question,
                        query=current_query,
                        evidence=deepcopy(evidence),
                    ),
                    source_name="search_web",
                )
            except RequestError as exc:
                steps.append(
                    InvestigationStep(
                        step_number=len(steps) + 1,
                        action=action,
                        reason=decision.reason,
                        query_before=query_before,
                        query_after=current_query,
                        added_evidence_count=0,
                        error_message=str(exc),
                        retryable=exc.retryable,
                    )
                )
                failed_decision = EvidenceDecision(
                    enough=False,
                    next_action=action,
                    reason=f"{decision.reason}；搜索未完成：{exc}",
                    next_query=decision.next_query,
                )
                return _build_result(
                    status="insufficient",
                    question=normalized_question,
                    final_query=current_query,
                    evidence=evidence,
                    steps=steps,
                    stop_reason="action_failed",
                    last_decision=failed_decision,
                )
        else:
            raise ValueError(f"不支持的调查动作: {action}")

        evidence, added_count = _merge_evidence(evidence, additions)
        steps.append(
            InvestigationStep(
                step_number=len(steps) + 1,
                action=action,
                reason=decision.reason,
                query_before=query_before,
                query_after=current_query,
                added_evidence_count=added_count,
            )
        )

        if added_count == 0:
            return _build_result(
                status="insufficient",
                question=normalized_question,
                final_query=current_query,
                evidence=evidence,
                steps=steps,
                stop_reason="no_progress",
                last_decision=decision,
            )
