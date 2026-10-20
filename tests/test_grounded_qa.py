from copy import deepcopy

import pytest
from pydantic import ValidationError

from errors import InvalidCitationError
from src.grounded_qa import (
    CitedClaim,
    GroundedAnswer,
    answer_with_citations,
)


def make_retrieval_results():
    return [
        {
            "doc_id": "report",
            "page_number": 28,
            "chunk_id": "report:p28:c0",
            "content_type": "text",
            "text": "Boomplay覆盖多个非洲国家。",
        },
        {
            "doc_id": "report",
            "page_number": 28,
            "chunk_id": "report:p28:t0:c0",
            "content_type": "table",
            "table_id": "report:p28:t0",
            "text": "| APP | 使用时长 |\n|---|---|\n| Boomplay | 40.78分钟 |",
        },
    ]


def test_cited_claim_normalizes_fields_and_deduplicates_sources():
    claim = CitedClaim(
        claim="  Boomplay为40.78分钟。  ",
        source_ids=[" S2 ", "S2"],
    )

    assert claim.claim == "Boomplay为40.78分钟。"
    assert claim.source_ids == ["S2"]


@pytest.mark.parametrize(
    "payload",
    [
        {
            "status": "answered",
            "claims": [],
            "missing_information": [],
        },
        {
            "status": "insufficient",
            "claims": [
                {"claim": "不应出现", "source_ids": ["S1"]}
            ],
            "missing_information": ["缺少数据"],
        },
        {
            "status": "insufficient",
            "claims": [],
            "missing_information": [],
        },
        {
            "status": "conflict",
            "claims": [],
            "missing_information": ["需要复核"],
        },
    ],
)
def test_grounded_answer_rejects_invalid_status_combinations(payload):
    with pytest.raises(ValidationError):
        GroundedAnswer.model_validate(payload)


def test_answer_with_citations_calls_generator_and_returns_mapping():
    retrieval_results = make_retrieval_results()
    original = deepcopy(retrieval_results)
    captured = {}

    def fake_generate_answer(*, system_prompt, user_prompt):
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        return {
            "status": "answered",
            "claims": [
                {
                    "claim": "Boomplay的使用时长为40.78分钟。",
                    "source_ids": ["S2"],
                }
            ],
            "missing_information": [],
        }

    answer, source_map = answer_with_citations(
        question="  Boomplay的使用时长是多少？  ",
        retrieval_results=retrieval_results,
        generate_answer=fake_generate_answer,
    )

    assert answer.status == "answered"
    assert answer.claims[0].source_ids == ["S2"]
    assert source_map["S2"]["table_id"] == "report:p28:t0"
    assert "Boomplay的使用时长是多少？" in captured["user_prompt"]
    assert "[S2]" in captured["user_prompt"]
    assert "只能依据" in captured["system_prompt"]
    assert retrieval_results == original


def test_answer_with_citations_supports_insufficient_answer():
    def fake_generate_answer(**_kwargs):
        return {
            "status": "insufficient",
            "claims": [],
            "missing_information": ["没有2025年的数据"],
        }

    answer, _ = answer_with_citations(
        "2025年的数值是多少？",
        make_retrieval_results(),
        fake_generate_answer,
    )

    assert answer.status == "insufficient"
    assert answer.missing_information == ["没有2025年的数据"]


def test_answer_with_citations_rejects_unknown_source_id():
    def fake_generate_answer(**_kwargs):
        return {
            "status": "answered",
            "claims": [
                {"claim": "错误引用", "source_ids": ["S8"]}
            ],
            "missing_information": [],
        }

    with pytest.raises(InvalidCitationError, match="S8"):
        answer_with_citations(
            "问题",
            make_retrieval_results(),
            fake_generate_answer,
        )


@pytest.mark.parametrize("question", [None, 123])
def test_answer_with_citations_rejects_non_string_question(question):
    with pytest.raises(TypeError, match="question"):
        answer_with_citations(
            question,
            make_retrieval_results(),
            lambda **_kwargs: {},
        )


def test_answer_with_citations_rejects_blank_question():
    with pytest.raises(ValueError, match="question"):
        answer_with_citations(
            "   ",
            make_retrieval_results(),
            lambda **_kwargs: {},
        )


def test_answer_with_citations_rejects_non_callable_generator():
    with pytest.raises(TypeError, match="generate_answer"):
        answer_with_citations(
            "问题",
            make_retrieval_results(),
            None,
        )
