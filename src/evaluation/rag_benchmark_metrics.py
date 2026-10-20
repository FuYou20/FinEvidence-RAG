"""面向研报 RAG 子任务的确定性评测指标。

检索指标只能说明证据是否被找到，不能直接代表最终业务答案正确。
本模块按题型分别统计，不生成一个掩盖问题的总分：

* single_fact：事实准确率、事实召回率；
* enumeration：项目级 Precision / Recall / F1、完整回答率；
* comparison：跨文档覆盖率、证据召回率，并接收外部语义裁判分；
* unanswerable：预期动作准确率、错误强答率。
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from numbers import Real


ANSWER_ACTION = "answer_with_citations"
VALID_ACTIONS = {
    ANSWER_ACTION,
    "abstain",
    "clarify",
    "external_research",
}
VALID_TASK_TYPES = {
    "single_fact",
    "enumeration",
    "comparison",
    "unanswerable",
}


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _normalize_text(value):
    if not isinstance(value, str):
        raise TypeError("待比较的文本必须是 str")
    value = unicodedata.normalize("NFKC", value).lower()
    # 百分数、数字、小数点和中英文保留，其余符号及空白不影响匹配。
    return re.sub(r"[^0-9a-z\u4e00-\u9fff%.]+", "", value)


def _fact_matches(answer_text, aliases):
    normalized_answer = _normalize_text(answer_text)
    return any(
        normalized_alias and normalized_alias in normalized_answer
        for normalized_alias in (_normalize_text(alias) for alias in aliases)
    )


def _required_doc_coverage(case, prediction):
    required = set(case.get("required_doc_ids", []))
    if not required:
        return 1.0
    cited = set(prediction["cited_doc_ids"])
    return len(required & cited) / len(required)


def _gold_chunk_ids(case):
    return {
        evidence["source_chunk_id"]
        for evidence in case.get("gold_evidence", [])
        if evidence.get("source_chunk_id")
    }


def _evidence_chunk_recall(case, prediction):
    gold = _gold_chunk_ids(case)
    if not gold:
        return 1.0
    cited = set(prediction["cited_chunk_ids"])
    return len(gold & cited) / len(gold)


def _validate_cases_and_predictions(cases, predictions):
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases 必须是非空 list")
    if not isinstance(predictions, list):
        raise TypeError("predictions 必须是 list")

    case_by_id = {}
    for case in cases:
        query_id = case.get("query_id")
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError("每条 case 必须具有非空 query_id")
        if query_id in case_by_id:
            raise ValueError(f"case query_id 重复: {query_id}")
        if case.get("task_type") not in VALID_TASK_TYPES:
            raise ValueError(f"不支持的 task_type: {case.get('task_type')}")
        case_by_id[query_id] = case

    prediction_by_id = {}
    required_fields = {
        "query_id",
        "predicted_action",
        "answer_text",
        "cited_chunk_ids",
        "cited_doc_ids",
        "predicted_items",
    }
    for prediction in predictions:
        missing = required_fields - prediction.keys()
        if missing:
            raise ValueError(f"prediction 缺少字段: {sorted(missing)}")
        query_id = prediction["query_id"]
        if query_id in prediction_by_id:
            raise ValueError(f"prediction query_id 重复: {query_id}")
        if prediction["predicted_action"] not in VALID_ACTIONS:
            raise ValueError(
                f"不支持的 predicted_action: {prediction['predicted_action']}"
            )
        if not isinstance(prediction["answer_text"], str):
            raise TypeError("answer_text 必须是 str")
        for field in ("cited_chunk_ids", "cited_doc_ids", "predicted_items"):
            if not isinstance(prediction[field], list):
                raise TypeError(f"{field} 必须是 list")
        prediction_by_id[query_id] = prediction

    missing_predictions = set(case_by_id) - set(prediction_by_id)
    extra_predictions = set(prediction_by_id) - set(case_by_id)
    if missing_predictions or extra_predictions:
        raise ValueError(
            "case 与 prediction 的 query_id 不一致: "
            f"missing={sorted(missing_predictions)}, "
            f"extra={sorted(extra_predictions)}"
        )
    return case_by_id, prediction_by_id


def _evaluate_single_fact(case, prediction):
    facts = case.get("gold_facts", [])
    if not facts:
        raise ValueError(f"single_fact 缺少 gold_facts: {case['query_id']}")
    if not all(isinstance(aliases, list) and aliases for aliases in facts):
        raise ValueError(f"gold_facts 必须是非空别名组列表: {case['query_id']}")

    matched_count = 0
    if prediction["predicted_action"] == ANSWER_ACTION:
        matched_count = sum(
            _fact_matches(prediction["answer_text"], aliases)
            for aliases in facts
        )
    fact_recall = matched_count / len(facts)
    return {
        "query_id": case["query_id"],
        "task_type": "single_fact",
        "accuracy": float(matched_count == len(facts)),
        "fact_recall": fact_recall,
        "required_doc_coverage": _required_doc_coverage(case, prediction),
        "evidence_chunk_recall": _evidence_chunk_recall(case, prediction),
    }


def _item_matches(predicted_text, gold_item):
    predicted = _normalize_text(predicted_text)
    if not predicted:
        return False
    candidates = [gold_item["item"], *gold_item.get("aliases", [])]
    for candidate in candidates:
        normalized_candidate = _normalize_text(candidate)
        if (
            normalized_candidate
            and (normalized_candidate in predicted or predicted in normalized_candidate)
        ):
            return True
    return False


def _match_items(gold_items, predicted_items, judged_matches=None):
    """贪心一对一匹配，避免一个预测项目重复命中多个 gold item。"""
    if judged_matches is not None:
        matches = []
        used_predicted_indexes = set()
        used_gold_indexes = set()
        for match in judged_matches:
            if not isinstance(match, dict):
                raise TypeError("enumeration_matches 的每一项必须是 dict")
            predicted_index = match.get("predicted_index")
            gold_index = match.get("gold_index")
            if (
                not isinstance(predicted_index, int)
                or isinstance(predicted_index, bool)
                or not 0 <= predicted_index < len(predicted_items)
            ):
                raise ValueError(f"无效 predicted_index: {predicted_index}")
            if (
                not isinstance(gold_index, int)
                or isinstance(gold_index, bool)
                or not 0 <= gold_index < len(gold_items)
            ):
                raise ValueError(f"无效 gold_index: {gold_index}")
            if predicted_index in used_predicted_indexes:
                raise ValueError(f"predicted_index 重复: {predicted_index}")
            if gold_index in used_gold_indexes:
                raise ValueError(f"gold_index 重复: {gold_index}")
            matches.append((predicted_index, gold_index))
            used_predicted_indexes.add(predicted_index)
            used_gold_indexes.add(gold_index)
        return matches

    used_gold_indexes = set()
    matches = []
    for predicted_index, predicted_item in enumerate(predicted_items):
        if not isinstance(predicted_item, dict):
            raise TypeError("predicted_items 的每一项必须是 dict")
        if not isinstance(predicted_item.get("item"), str):
            raise TypeError("predicted item 的 item 必须是 str")
        if not isinstance(predicted_item.get("cited_chunk_ids", []), list):
            raise TypeError("predicted item 的 cited_chunk_ids 必须是 list")
        for gold_index, gold_item in enumerate(gold_items):
            if gold_index in used_gold_indexes:
                continue
            if _item_matches(predicted_item["item"], gold_item):
                used_gold_indexes.add(gold_index)
                matches.append((predicted_index, gold_index))
                break
    return matches


def _evaluate_enumeration(case, prediction):
    gold_items = case.get("gold_items", [])
    if not gold_items:
        raise ValueError(f"enumeration 缺少 gold_items: {case['query_id']}")
    predicted_items = (
        prediction["predicted_items"]
        if prediction["predicted_action"] == ANSWER_ACTION
        else []
    )
    matches = _match_items(
        gold_items,
        predicted_items,
        judged_matches=prediction.get("enumeration_matches"),
    )
    true_positive = len(matches)
    precision = true_positive / len(predicted_items) if predicted_items else 0.0
    recall = true_positive / len(gold_items)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    correctly_cited = 0
    for predicted_index, gold_index in matches:
        predicted_citations = set(
            predicted_items[predicted_index].get("cited_chunk_ids", [])
        )
        gold_citations = set(gold_items[gold_index].get("evidence_chunk_ids", []))
        correctly_cited += bool(predicted_citations & gold_citations)

    return {
        "query_id": case["query_id"],
        "task_type": "enumeration",
        "item_precision": precision,
        "item_recall": recall,
        "item_f1": f1,
        "complete_answer": float(
            true_positive == len(gold_items)
            and len(predicted_items) == len(gold_items)
        ),
        "item_citation_accuracy": (
            correctly_cited / true_positive if true_positive else 0.0
        ),
        "required_doc_coverage": _required_doc_coverage(case, prediction),
        "evidence_chunk_recall": _evidence_chunk_recall(case, prediction),
    }


def _evaluate_comparison(case, prediction):
    row = {
        "query_id": case["query_id"],
        "task_type": "comparison",
        "answered": float(prediction["predicted_action"] == ANSWER_ACTION),
        "required_doc_coverage": _required_doc_coverage(case, prediction),
        "evidence_chunk_recall": _evidence_chunk_recall(case, prediction),
        "comparison_score": None,
    }
    score = prediction.get("comparison_score")
    if score is not None:
        if isinstance(score, bool) or not isinstance(score, Real):
            raise TypeError("comparison_score 必须是 0 到 1 的数值")
        if not 0 <= score <= 1:
            raise ValueError("comparison_score 必须位于 [0, 1]")
        row["comparison_score"] = float(score)
    return row


def _evaluate_unanswerable(case, prediction):
    expected_action = case["expected_action"]
    predicted_action = prediction["predicted_action"]
    return {
        "query_id": case["query_id"],
        "task_type": "unanswerable",
        "expected_action": expected_action,
        "predicted_action": predicted_action,
        "action_accuracy": float(predicted_action == expected_action),
        "unsupported_answer": float(predicted_action == ANSWER_ACTION),
    }


def _aggregate(rows, keys):
    result = {"query_count": len(rows)}
    for key in keys:
        result[key] = _mean([row[key] for row in rows])
    return result


def evaluate_rag_benchmark(cases, predictions):
    """按子任务评测一组端到端 RAG 预测，不计算误导性的统一总分。"""
    case_by_id, prediction_by_id = _validate_cases_and_predictions(
        cases, predictions
    )
    per_query = []
    grouped = defaultdict(list)

    evaluators = {
        "single_fact": _evaluate_single_fact,
        "enumeration": _evaluate_enumeration,
        "comparison": _evaluate_comparison,
        "unanswerable": _evaluate_unanswerable,
    }
    for query_id, case in case_by_id.items():
        row = evaluators[case["task_type"]](case, prediction_by_id[query_id])
        per_query.append(row)
        grouped[case["task_type"]].append(row)

    comparison_scores = [
        row["comparison_score"]
        for row in grouped["comparison"]
        if row["comparison_score"] is not None
    ]
    expected_action_stats = {}
    for expected_action in ("abstain", "clarify", "external_research"):
        rows = [
            row
            for row in grouped["unanswerable"]
            if row["expected_action"] == expected_action
        ]
        expected_action_stats[expected_action] = {
            "query_count": len(rows),
            "accuracy": _mean([row["action_accuracy"] for row in rows]),
        }

    return {
        "query_count": len(cases),
        "single_fact": _aggregate(
            grouped["single_fact"],
            (
                "accuracy",
                "fact_recall",
                "required_doc_coverage",
                "evidence_chunk_recall",
            ),
        ),
        "enumeration": _aggregate(
            grouped["enumeration"],
            (
                "item_precision",
                "item_recall",
                "item_f1",
                "complete_answer",
                "item_citation_accuracy",
                "required_doc_coverage",
                "evidence_chunk_recall",
            ),
        ),
        "comparison": {
            **_aggregate(
                grouped["comparison"],
                (
                    "answered",
                    "required_doc_coverage",
                    "evidence_chunk_recall",
                ),
            ),
            "semantic_judged_count": len(comparison_scores),
            "comparison_score": (
                _mean(comparison_scores) if comparison_scores else None
            ),
        },
        "unanswerable": {
            **_aggregate(
                grouped["unanswerable"],
                ("action_accuracy", "unsupported_answer"),
            ),
            "by_expected_action": expected_action_stats,
        },
        "per_query": per_query,
    }
