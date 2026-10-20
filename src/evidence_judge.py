from .evidence_context import build_evidence_context
from .context_budget import (
    DEFAULT_MAX_EVIDENCE_ITEMS,
    DEFAULT_MAX_TEXT_CHARS_PER_ITEM,
    DEFAULT_MAX_TOTAL_TEXT_CHARS,
    compact_evidence_records,
)
from .investigation_schema import EvidenceDecision


SYSTEM_PROMPT = """你是中立证据调查工作流中的证据判断器。
你的职责只有两个：判断现有材料在充分性与真实性方面是否达到交付要求；如果不能，只选择一个下一步动作。

充分性要求：材料应覆盖问题所需的对象、时间、指标、数值、单位和关键角度，不得因为内容相关就提前结束。
真实性要求：关键证据应能追溯来源，引用内容与原文一致，时效适合当前问题；必要时应有独立来源交叉验证。

可选动作：
- finish：证据包已经达到当前调查任务所需的覆盖度与可验证程度，可以交给上层 Agent。
- rewrite_query：没有找到相关证据，或当前检索表达可能没有命中文档措辞。
- multimodal：相关页面提到图片、图、图表或曲线，但关键内容没有出现在文本或表格证据中。
- web_search：问题要求当前信息、来源不可追溯、需要独立来源验证，或相同时间和口径的证据存在冲突。

规则：
1. 一次只能选择一个动作，不得要求同时执行多个动作。
2. 选择 rewrite_query 或 web_search 时，必须直接给出可执行的 next_query；选择 finish 或 multimodal 时 next_query 必须为 null。
3. 证据不足时不得选择 finish。
4. 不得判断投资观点是否正确，不得替正反方组织论点或决定胜负。
5. 不得利用证据之外的知识直接回答问题。
6. 不要因为可以联网就默认选择 web_search。
7. source_kind=web 且 content_origin=search_summary 的材料只是搜索摘要，不能声称已读取网页全文或已核实原文；缺少足够原文依据时明确说明限制。

示例：
- 没有检索到相关内容 -> rewrite_query，并在 next_query 中给出包含关键实体、时间和指标的新检索式。
- 正文写“具体份额见图表27”，但没有图中数值 -> multimodal。
- 本地研报是2022年，问题询问“目前”的市场份额 -> web_search，并在 next_query 中给出适合网页搜索的检索式。
- 两个来源对同一年度、同一口径给出不同数值 -> web_search。
- 现有证据已经包含对象、时间、指标、数值和来源 -> finish。"""


def _normalize_required_string(value, *, field_name):
    if not isinstance(value, str):
        raise TypeError(f"{field_name} 必须是字符串")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} 不能为空")
    return normalized


def evaluate_evidence(
    question,
    evidence,
    generate_decision,
    *,
    query=None,
    available_actions=None,
    max_evidence_items=DEFAULT_MAX_EVIDENCE_ITEMS,
    max_evidence_text_chars=DEFAULT_MAX_TOTAL_TEXT_CHARS,
    max_chars_per_evidence=DEFAULT_MAX_TEXT_CHARS_PER_ITEM,
):
    """调用结构化判断器，决定结束调查或选择一个深入调查动作。"""
    normalized_question = _normalize_required_string(
        question,
        field_name="question",
    )
    normalized_query = _normalize_required_string(
        query if query is not None else normalized_question,
        field_name="query",
    )
    if not isinstance(evidence, list):
        raise TypeError("evidence 必须是 list")
    if not callable(generate_decision):
        raise TypeError("generate_decision 必须是可调用对象")

    action_hint = ""
    if available_actions is not None:
        if not isinstance(available_actions, (set, list, tuple)):
            raise TypeError("available_actions 必须是 set、list 或 tuple")
        allowed = {"finish", "rewrite_query", "multimodal", "web_search"}
        normalized_actions = []
        for action in available_actions:
            if action not in allowed:
                raise ValueError(f"不支持的可用动作: {action}")
            if action not in normalized_actions:
                normalized_actions.append(action)
        if "finish" not in normalized_actions:
            normalized_actions.insert(0, "finish")
        action_hint = (
            "\n\n当前实际可用动作："
            + "、".join(normalized_actions)
            + "。不得选择未列出的动作。"
        )

    if evidence:
        prompt_evidence = compact_evidence_records(
            evidence,
            max_items=max_evidence_items,
            max_total_text_chars=max_evidence_text_chars,
            max_text_chars_per_item=max_chars_per_evidence,
        )
        evidence_context, _ = build_evidence_context(prompt_evidence)
    else:
        evidence_context = "当前没有检索到任何证据。"

    user_prompt = (
        f"原始问题：\n{normalized_question}\n\n"
        f"当前检索 Query：\n{normalized_query}\n\n"
        f"当前证据：\n{evidence_context}"
        f"{action_hint}"
    )
    raw_decision = generate_decision(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    return EvidenceDecision.model_validate(raw_decision)
