from copy import deepcopy

import pytest

from errors import RequestError
from src.evidence_investigation import run_evidence_investigation


def make_evidence(chunk_id, text=None):
    return {
        "doc_id": "report_001",
        "page_number": 1,
        "chunk_id": chunk_id,
        "content_type": "text",
        "text": text or chunk_id,
    }


def test_workflow_finishes_after_initial_retrieval():
    evidence = [make_evidence("c1")]
    original = deepcopy(evidence)
    calls = {"retrieve": 0, "judge": 0}

    def retrieve_evidence(*, query):
        calls["retrieve"] += 1
        assert query == "传音的优势是什么？"
        return evidence

    def judge_evidence(**kwargs):
        calls["judge"] += 1
        assert kwargs["evidence"] == evidence
        return {
            "enough": True,
            "next_action": "finish",
            "reason": "证据充分",
        }

    result = run_evidence_investigation(
        "  传音的优势是什么？  ",
        retrieve_evidence,
        judge_evidence,
    )

    assert result.status == "sufficient"
    assert result.stop_reason == "evidence_sufficient"
    assert result.steps == []
    assert calls == {"retrieve": 1, "judge": 1}
    assert evidence == original


def test_workflow_can_start_from_registered_initial_evidence():
    initial = [make_evidence("registered")]
    retrieve_calls = []

    result = run_evidence_investigation(
        "核对指定证据",
        lambda **kwargs: retrieve_calls.append(kwargs),
        lambda **_kwargs: {
            "enough": True,
            "next_action": "finish",
            "reason": "原始证据已成功还原",
        },
        initial_evidence=initial,
    )

    assert result.evidence == initial
    assert retrieve_calls == []


def test_workflow_rewrites_query_and_retrieves_again():
    queries = []
    decisions = iter(
        [
            {
                "enough": False,
                "next_action": "rewrite_query",
                "reason": "原查询没有命中文档措辞",
                "next_query": "本土化运营 渠道优势",
            },
            {
                "enough": True,
                "next_action": "finish",
                "reason": "新查询找到了直接证据",
            },
        ]
    )

    def retrieve_evidence(*, query):
        queries.append(query)
        if query == "本土化运营 渠道优势":
            return [make_evidence("c2")]
        return []

    result = run_evidence_investigation(
        "传音为什么有竞争优势？",
        retrieve_evidence,
        lambda **_kwargs: next(decisions),
    )

    assert result.status == "sufficient"
    assert queries == [
        "传音为什么有竞争优势？",
        "本土化运营 渠道优势",
    ]
    assert result.final_query == "本土化运营 渠道优势"
    assert result.steps[0].action == "rewrite_query"
    assert result.steps[0].added_evidence_count == 1


def test_workflow_only_runs_selected_multimodal_action():
    calls = {"multimodal": 0, "web": 0}
    decisions = iter(
        [
            {
                "enough": False,
                "next_action": "multimodal",
                "reason": "正文指向图表27",
            },
            {
                "enough": True,
                "next_action": "finish",
                "reason": "已取得图表数值",
            },
        ]
    )

    def parse_multimodal(**_kwargs):
        calls["multimodal"] += 1
        return [make_evidence("figure-27", "图表数值为40%")]

    def search_web(**_kwargs):
        calls["web"] += 1
        return [make_evidence("web")]

    result = run_evidence_investigation(
        "市场份额是多少？",
        lambda **_kwargs: [make_evidence("c1", "见图表27")],
        lambda **_kwargs: next(decisions),
        parse_multimodal=parse_multimodal,
        search_web=search_web,
    )

    assert result.status == "sufficient"
    assert calls == {"multimodal": 1, "web": 0}
    assert len(result.evidence) == 2


def test_workflow_runs_web_search_for_cross_validation():
    decisions = iter(
        [
            {
                "enough": False,
                "next_action": "web_search",
                "reason": "本地研报已经过时",
                "next_query": "传音 非洲 市场份额 最新",
            },
            {
                "enough": True,
                "next_action": "finish",
                "reason": "已经获得当前数据",
            },
        ]
    )

    result = run_evidence_investigation(
        "目前的市场份额是多少？",
        lambda **_kwargs: [make_evidence("old")],
        lambda **_kwargs: next(decisions),
        search_web=lambda **_kwargs: [make_evidence("current")],
    )

    assert result.status == "sufficient"
    assert result.steps[0].action == "web_search"


def test_workflow_stops_when_selected_action_is_unavailable():
    result = run_evidence_investigation(
        "需要解析图表",
        lambda **_kwargs: [make_evidence("c1")],
        lambda **_kwargs: {
            "enough": False,
            "next_action": "multimodal",
            "reason": "需要图表数据",
        },
    )

    assert result.status == "insufficient"
    assert result.stop_reason == "action_unavailable"
    assert result.steps == []


def test_workflow_stops_when_action_adds_no_new_evidence():
    original = make_evidence("c1")
    result = run_evidence_investigation(
        "需要外部验证",
        lambda **_kwargs: [original],
        lambda **_kwargs: {
            "enough": False,
            "next_action": "web_search",
            "reason": "需要独立来源",
            "next_query": "独立来源 验证",
        },
        search_web=lambda **_kwargs: [deepcopy(original)],
    )

    assert result.status == "insufficient"
    assert result.stop_reason == "no_progress"
    assert result.steps[0].added_evidence_count == 0
    assert len(result.evidence) == 1


def test_search_failure_preserves_partial_evidence_and_failure_reason():
    def unavailable_search(**_kwargs):
        raise RequestError("博查权限不足", retryable=False, status_code=401)

    result = run_evidence_investigation(
        "检查最新数据",
        lambda **_kwargs: [make_evidence("old")],
        lambda **_kwargs: {
            "enough": False,
            "next_action": "web_search",
            "reason": "旧材料需要独立核验",
            "next_query": "最新数据 独立来源",
        },
        search_web=unavailable_search,
    )

    assert result.status == "insufficient"
    assert result.stop_reason == "action_failed"
    assert len(result.evidence) == 1
    assert result.steps[0].retryable is False
    assert "权限不足" in result.last_decision.reason


def test_workflow_stops_after_maximum_deepening_steps():
    rewrite_number = 0

    def judge_evidence(**_kwargs):
        nonlocal rewrite_number
        rewrite_number += 1
        return {
            "enough": False,
            "next_action": "rewrite_query",
            "reason": "仍需扩大检索",
            "next_query": f"改写查询 {rewrite_number}",
        }

    def retrieve_evidence(*, query):
        return [make_evidence(query)]

    result = run_evidence_investigation(
        "原问题",
        retrieve_evidence,
        judge_evidence,
        max_deepening_steps=2,
    )

    assert result.status == "insufficient"
    assert result.stop_reason == "max_deepening_steps"
    assert len(result.steps) == 2
    assert len(result.evidence) == 3


def test_seed_evidence_is_preserved_and_merged_with_fresh_retrieval():
    seed = make_evidence("original", "原观点已经引用的证据")
    fresh = make_evidence("fresh", "围绕质疑重新检索的新证据")

    result = run_evidence_investigation(
        "重新调查观点",
        lambda **_kwargs: [fresh],
        lambda **_kwargs: {
            "enough": True,
            "next_action": "finish",
            "reason": "原证据和新证据均已保留",
        },
        seed_evidence=[seed],
    )

    assert [item["chunk_id"] for item in result.evidence] == [
        "original",
        "fresh",
    ]


@pytest.mark.parametrize("max_steps", [-1, 1.5, True, "2"])
def test_workflow_rejects_invalid_max_steps(max_steps):
    with pytest.raises(ValueError, match="max_deepening_steps"):
        run_evidence_investigation(
            "问题",
            lambda **_kwargs: [],
            lambda **_kwargs: {},
            max_deepening_steps=max_steps,
        )


def test_workflow_rejects_non_list_retrieval_result():
    with pytest.raises(TypeError, match="retrieve_evidence 必须返回 list"):
        run_evidence_investigation(
            "问题",
            lambda **_kwargs: {},
            lambda **_kwargs: {},
        )
