from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from errors import LLMJsonParseError
from src.grounded_qa import GroundedAnswer
from src.investigation_schema import EvidenceDecision
from src.llm_generator import (
    create_evidence_decision_generator,
    create_grounded_answer_generator,
    create_text_generator,
)


class FakeResponses:
    def __init__(self, *, output_text="模型生成的回答。[S1]", error=None):
        self.output_text = output_text
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(output_text=self.output_text)


class FakeClient:
    def __init__(self, *, output_text="模型生成的回答。[S1]", error=None):
        self.responses = FakeResponses(
            output_text=output_text,
            error=error,
        )


class FakeStructuredResponses:
    def __init__(self, *, output_parsed=None, error=None):
        self.output_parsed = output_parsed
        self.error = error
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(output_parsed=self.output_parsed)


class FakeStructuredClient:
    def __init__(self, *, output_parsed=None, error=None):
        self.responses = FakeStructuredResponses(
            output_parsed=output_parsed,
            error=error,
        )


def test_create_text_generator_returns_callable():
    generator = create_text_generator(
        FakeClient(),
        model_name="test-model",
    )

    assert callable(generator)


def test_generator_calls_responses_api_and_returns_text():
    client = FakeClient(output_text="模型生成的回答。[S1]")
    generator = create_text_generator(
        client,
        model_name="  test-model  ",
    )

    result = generator(
        system_prompt="  你是研究助手  ",
        user_prompt="  传音有哪些竞争优势？  ",
    )

    assert result == "模型生成的回答。[S1]"
    assert len(client.responses.calls) == 1

    request = client.responses.calls[0]
    assert request["model"] == "test-model"
    assert request["input"] == [
        {
            "role": "system",
            "content": "你是研究助手",
        },
        {
            "role": "user",
            "content": "传音有哪些竞争优势？",
        },
    ]


@pytest.mark.parametrize("model_name", [None, 123, [], {}])
def test_create_text_generator_rejects_non_string_model_name(model_name):
    with pytest.raises(TypeError, match="model_name 必须是字符串"):
        create_text_generator(
            FakeClient(),
            model_name=model_name,
        )


@pytest.mark.parametrize("model_name", ["", "   ", "\n\t"])
def test_create_text_generator_rejects_blank_model_name(model_name):
    with pytest.raises(ValueError, match="model_name 不能为空"):
        create_text_generator(
            FakeClient(),
            model_name=model_name,
        )


@pytest.mark.parametrize("system_prompt", [None, 123, [], {}])
def test_generator_rejects_non_string_system_prompt(system_prompt):
    generator = create_text_generator(
        FakeClient(),
        model_name="test-model",
    )

    with pytest.raises(TypeError, match="system_prompt 必须是字符串"):
        generator(
            system_prompt=system_prompt,
            user_prompt="用户问题",
        )


@pytest.mark.parametrize("system_prompt", ["", "   ", "\n\t"])
def test_generator_rejects_blank_system_prompt(system_prompt):
    generator = create_text_generator(
        FakeClient(),
        model_name="test-model",
    )

    with pytest.raises(ValueError, match="system_prompt 不能为空"):
        generator(
            system_prompt=system_prompt,
            user_prompt="用户问题",
        )


@pytest.mark.parametrize("user_prompt", [None, 123, [], {}])
def test_generator_rejects_non_string_user_prompt(user_prompt):
    generator = create_text_generator(
        FakeClient(),
        model_name="test-model",
    )

    with pytest.raises(TypeError, match="user_prompt 必须是字符串"):
        generator(
            system_prompt="系统提示",
            user_prompt=user_prompt,
        )


@pytest.mark.parametrize("user_prompt", ["", "   ", "\n\t"])
def test_generator_rejects_blank_user_prompt(user_prompt):
    generator = create_text_generator(
        FakeClient(),
        model_name="test-model",
    )

    with pytest.raises(ValueError, match="user_prompt 不能为空"):
        generator(
            system_prompt="系统提示",
            user_prompt=user_prompt,
        )


def test_generator_propagates_sdk_exception_unchanged():
    provider_error = RuntimeError("模型服务暂时不可用")
    client = FakeClient(error=provider_error)
    generator = create_text_generator(
        client,
        model_name="test-model",
    )

    with pytest.raises(RuntimeError) as exc_info:
        generator(
            system_prompt="系统提示",
            user_prompt="用户问题",
        )

    assert exc_info.value is provider_error
    assert len(client.responses.calls) == 1


def make_answer_data():
    return {
        "status": "answered",
        "claims": [
            {
                "claim": "传音通过本土化运营建立竞争优势。",
                "source_ids": ["S1"],
            }
        ],
        "missing_information": [],
    }


def test_create_grounded_answer_generator_returns_callable():
    generator = create_grounded_answer_generator(
        FakeStructuredClient(),
        model_name="test-model",
    )

    assert callable(generator)


def test_grounded_generator_calls_parse_and_returns_model():
    client = FakeStructuredClient(output_parsed=make_answer_data())
    generator = create_grounded_answer_generator(
        client,
        model_name="  test-model  ",
    )

    result = generator(
        system_prompt="  你是证据驱动的研究助手  ",
        user_prompt="  传音有哪些竞争优势？  ",
    )

    assert isinstance(result, GroundedAnswer)
    assert result.status == "answered"
    assert result.claims[0].source_ids == ["S1"]
    assert len(client.responses.calls) == 1

    request = client.responses.calls[0]
    assert request["model"] == "test-model"
    assert request["text_format"] is GroundedAnswer
    assert request["input"] == [
        {
            "role": "system",
            "content": "你是证据驱动的研究助手",
        },
        {
            "role": "user",
            "content": "传音有哪些竞争优势？",
        },
    ]


def test_grounded_generator_accepts_already_parsed_model():
    parsed_answer = GroundedAnswer.model_validate(make_answer_data())
    generator = create_grounded_answer_generator(
        FakeStructuredClient(output_parsed=parsed_answer),
        model_name="test-model",
    )

    result = generator(
        system_prompt="系统提示",
        user_prompt="用户问题",
    )

    assert result is parsed_answer


def test_evidence_decision_generator_calls_parse_and_returns_model():
    client = FakeStructuredClient(
        output_parsed={
            "enough": False,
            "next_action": "multimodal",
            "reason": "关键数值位于图表中",
        }
    )
    generator = create_evidence_decision_generator(
        client,
        model_name="test-model",
    )

    result = generator(
        system_prompt="你是证据判断器",
        user_prompt="判断当前证据",
    )

    assert isinstance(result, EvidenceDecision)
    assert result.next_action == "multimodal"
    request = client.responses.calls[0]
    assert request["text_format"] is EvidenceDecision


def test_evidence_decision_generator_rejects_missing_parsed_output():
    generator = create_evidence_decision_generator(
        FakeStructuredClient(output_parsed=None),
        model_name="test-model",
    )

    with pytest.raises(LLMJsonParseError, match="EvidenceDecision"):
        generator(
            system_prompt="系统提示",
            user_prompt="用户提示",
        )


@pytest.mark.parametrize("model_name", [None, 123, [], {}])
def test_grounded_generator_rejects_non_string_model_name(model_name):
    with pytest.raises(TypeError, match="model_name 必须是字符串"):
        create_grounded_answer_generator(
            FakeStructuredClient(),
            model_name=model_name,
        )


@pytest.mark.parametrize("model_name", ["", "   ", "\n\t"])
def test_grounded_generator_rejects_blank_model_name(model_name):
    with pytest.raises(ValueError, match="model_name 不能为空"):
        create_grounded_answer_generator(
            FakeStructuredClient(),
            model_name=model_name,
        )


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("system_prompt", None),
        ("system_prompt", "   "),
        ("user_prompt", 123),
        ("user_prompt", "\n\t"),
    ],
)
def test_grounded_generator_rejects_invalid_prompts(
    field_name,
    field_value,
):
    generator = create_grounded_answer_generator(
        FakeStructuredClient(output_parsed=make_answer_data()),
        model_name="test-model",
    )
    prompts = {
        "system_prompt": "系统提示",
        "user_prompt": "用户问题",
    }
    prompts[field_name] = field_value

    expected_error = TypeError if not isinstance(field_value, str) else ValueError
    with pytest.raises(expected_error, match=field_name):
        generator(**prompts)


def test_grounded_generator_rejects_missing_parsed_output():
    generator = create_grounded_answer_generator(
        FakeStructuredClient(output_parsed=None),
        model_name="test-model",
    )

    with pytest.raises(LLMJsonParseError, match="未返回可解析"):
        generator(
            system_prompt="系统提示",
            user_prompt="用户问题",
        )


def test_grounded_generator_wraps_invalid_parsed_output():
    invalid_answer = {
        "status": "answered",
        "claims": [],
        "missing_information": [],
    }
    generator = create_grounded_answer_generator(
        FakeStructuredClient(output_parsed=invalid_answer),
        model_name="test-model",
    )

    with pytest.raises(
        LLMJsonParseError,
        match="不符合 GroundedAnswer 协议",
    ) as exc_info:
        generator(
            system_prompt="系统提示",
            user_prompt="用户问题",
        )

    assert isinstance(exc_info.value.__cause__, ValidationError)


def test_grounded_generator_propagates_sdk_exception_unchanged():
    provider_error = RuntimeError("结构化输出接口暂时不可用")
    client = FakeStructuredClient(error=provider_error)
    generator = create_grounded_answer_generator(
        client,
        model_name="test-model",
    )

    with pytest.raises(RuntimeError) as exc_info:
        generator(
            system_prompt="系统提示",
            user_prompt="用户问题",
        )

    assert exc_info.value is provider_error
    assert len(client.responses.calls) == 1
