from src.evidence_dispatcher import dispatch_evidence_request
from src.evidence_investigation import run_evidence_investigation
from src.evidence_verification import handle_verify_evidence
from src.stance_research import handle_stance_research


def make_runner(
    *,
    retrieve_evidence,
    judge_evidence,
    rewrite_query=None,
    parse_multimodal=None,
    search_web=None,
):
    """把底层调查依赖绑定成处理器需要的统一入口。"""

    def investigation_runner(*, question, max_deepening_steps):
        return run_evidence_investigation(
            question=question,
            retrieve_evidence=retrieve_evidence,
            judge_evidence=judge_evidence,
            rewrite_query=rewrite_query,
            parse_multimodal=parse_multimodal,
            search_web=search_web,
            max_deepening_steps=max_deepening_steps,
        )

    return investigation_runner


def dispatch(request, runner):
    """使用同一个调查入口组装两种中立证据处理器。"""
    return dispatch_evidence_request(
        request=request,
        stance_research=lambda *, request: handle_stance_research(
            request,
            runner,
        ),
        verify_evidence=lambda *, request: handle_verify_evidence(
            request,
            runner,
        ),
    )


def test_stance_research_finishes_when_initial_evidence_is_complete():
    evidence = [
        {
            "evidence_id": "E1",
            "text": "传音采用多品牌策略覆盖不同消费层级。",
            "source": "2022年公司研报第15页",
        },
        {
            "evidence_id": "E2",
            "text": "传音在非洲长期进行本土化运营。",
            "source": "2022年公司研报第20页",
        },
    ]
    runner = make_runner(
        retrieve_evidence=lambda *, query: evidence,
        judge_evidence=lambda **_kwargs: {
            "enough": True,
            "next_action": "finish",
            "reason": "品牌与本土化两个调查重点均有可追溯材料",
        },
    )

    bundle = dispatch(
        {
            "request_id": "stance-complete",
            "requester": "supporting_agent",
            "mode": "stance_research",
            "topic": "传音的非洲市场优势是否具有持续性",
            "direction": "support",
            "research_focus": ["品牌矩阵", "本土化运营"],
        },
        runner,
    )

    assert bundle.status == "sufficient"
    assert [item["evidence_id"] for item in bundle.evidence] == [
        "E1",
        "E2",
    ]
    assert bundle.investigation_trace[-1]["stop_reason"] == (
        "evidence_sufficient"
    )
    assert len(bundle.investigation_trace) == 1


def test_stance_research_rewrites_query_to_fill_missing_angle():
    calls = {"retrieve": 0, "rewrite": 0}

    def retrieve_evidence(*, query):
        calls["retrieve"] += 1
        if query == "传音 非洲 渠道 品牌 本土化":
            return [
                {
                    "evidence_id": "E2",
                    "text": "销售网络覆盖多个非洲国家和地区。",
                    "source": "公司研报第20页",
                }
            ]
        return [
            {
                "evidence_id": "E1",
                "text": "传音采用多品牌策略。",
                "source": "公司研报第15页",
            }
        ]

    def judge_evidence(*, evidence, **_kwargs):
        if len(evidence) == 1:
            return {
                "enough": False,
                "next_action": "rewrite_query",
                "reason": "只有品牌材料，缺少渠道与本土化信息",
                "next_query": "传音 非洲 渠道 品牌 本土化",
            }
        return {
            "enough": True,
            "next_action": "finish",
            "reason": "调查重点已经得到覆盖",
        }

    def rewrite_query(**_kwargs):
        calls["rewrite"] += 1
        return "传音 非洲 渠道 品牌 本土化"

    runner = make_runner(
        retrieve_evidence=retrieve_evidence,
        judge_evidence=judge_evidence,
        rewrite_query=rewrite_query,
    )

    bundle = dispatch(
        {
            "request_id": "stance-rewrite",
            "requester": "supporting_agent",
            "mode": "stance_research",
            "topic": "传音在非洲的竞争优势",
            "direction": "support",
            "research_focus": ["品牌", "渠道"],
            "max_deepening_steps": 2,
        },
        runner,
    )

    assert bundle.status == "sufficient"
    assert calls == {"retrieve": 2, "rewrite": 0}
    assert len(bundle.evidence) == 2
    assert bundle.investigation_trace[0]["action"] == "rewrite_query"
    assert bundle.investigation_trace[0]["added_evidence_count"] == 1


def test_verify_evidence_uses_web_for_outdated_source():
    old_evidence = {
        "evidence_id": "E-old",
        "text": "2022年传音在非洲智能手机市场排名第一。",
        "published_at": "2022-04-22",
        "source": "历史研报",
    }
    new_evidence = {
        "evidence_id": "E-new",
        "text": "2026年独立机构发布了最新同口径市场数据。",
        "published_at": "2026-08-01",
        "source": "独立行业机构",
    }

    def judge_evidence(*, evidence, **_kwargs):
        if not any(item["evidence_id"] == "E-new" for item in evidence):
            return {
                "enough": False,
                "next_action": "web_search",
                "reason": "原证据时间过旧，需要最新独立来源",
                "next_query": "最新 独立来源",
            }
        return {
            "enough": True,
            "next_action": "finish",
            "reason": "原证据和最新独立来源均已取得",
        }

    runner = make_runner(
        retrieve_evidence=lambda *, query: [old_evidence],
        judge_evidence=judge_evidence,
        search_web=lambda **_kwargs: [new_evidence],
    )

    bundle = dispatch(
        {
            "request_id": "verify-outdated",
            "requester": "opposing_agent",
            "mode": "verify_evidence",
            "evidence_ids": ["E-old"],
            "verification_goal": "检查时效性并寻找最新独立来源",
            "max_deepening_steps": 2,
        },
        runner,
    )

    assert bundle.status == "sufficient"
    assert [item["evidence_id"] for item in bundle.evidence] == [
        "E-old",
        "E-new",
    ]
    assert bundle.investigation_trace[0]["action"] == "web_search"
    assert "supported" not in bundle.model_dump_json()
    assert "refuted" not in bundle.model_dump_json()


def test_verify_evidence_uses_multimodal_for_missing_chart_values():
    text_evidence = {
        "evidence_id": "E-text",
        "text": "具体市场份额见图表27。",
        "source": "研报第15页",
    }
    chart_evidence = {
        "evidence_id": "E-chart",
        "text": "图表27显示对应年份与市场份额数值。",
        "content_type": "figure",
        "source": "研报第15页图表27",
    }

    def judge_evidence(*, evidence, **_kwargs):
        if not any(
            item.get("content_type") == "figure" for item in evidence
        ):
            return {
                "enough": False,
                "next_action": "multimodal",
                "reason": "正文指向图表，但缺少图中的关键数值",
            }
        return {
            "enough": True,
            "next_action": "finish",
            "reason": "图表中的年份与数值已经补齐",
        }

    runner = make_runner(
        retrieve_evidence=lambda *, query: [text_evidence],
        judge_evidence=judge_evidence,
        parse_multimodal=lambda **_kwargs: [chart_evidence],
    )

    bundle = dispatch(
        {
            "request_id": "verify-chart",
            "requester": "opposing_agent",
            "mode": "verify_evidence",
            "evidence_ids": ["E-text"],
            "verification_goal": "核对图表中的年份和市场份额数值",
        },
        runner,
    )

    assert bundle.status == "sufficient"
    assert bundle.evidence[-1]["evidence_id"] == "E-chart"
    assert bundle.investigation_trace[0]["action"] == "multimodal"


def test_verify_evidence_returns_partial_material_when_no_progress():
    old_evidence = {
        "evidence_id": "E-only",
        "text": "无法确认原始出处的市场份额转述。",
        "source": "二次转载",
    }
    runner = make_runner(
        retrieve_evidence=lambda *, query: [old_evidence],
        judge_evidence=lambda **_kwargs: {
            "enough": False,
            "next_action": "web_search",
            "reason": "尚未找到原始来源或独立验证材料",
            "next_query": "原始来源 独立验证",
        },
        search_web=lambda **_kwargs: [old_evidence],
    )

    bundle = dispatch(
        {
            "request_id": "verify-no-progress",
            "requester": "judge_agent",
            "mode": "verify_evidence",
            "evidence_ids": ["E-only"],
            "verification_goal": "定位原始来源并交叉验证",
            "max_deepening_steps": 2,
        },
        runner,
    )

    assert bundle.status == "insufficient"
    assert bundle.evidence == [old_evidence]
    assert bundle.missing_information == [
        "尚未找到原始来源或独立验证材料"
    ]
    assert bundle.investigation_trace[0]["action"] == "web_search"
    assert bundle.investigation_trace[-1]["stop_reason"] == "no_progress"
