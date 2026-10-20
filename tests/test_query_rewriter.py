from types import SimpleNamespace

import pytest

from errors import LLMJsonParseError
from src.query_rewriter import RewrittenQuery, create_query_rewriter


class FakeResponses:
    def __init__(self, output_parsed):
        self.output_parsed = output_parsed
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_parsed=self.output_parsed)


def test_query_rewriter_uses_structured_output_and_returns_query():
    responses = FakeResponses(
        {
            "query": "传音 非洲 本土化 渠道 竞争壁垒",
            "rationale": "原查询缺少研报常用措辞",
        }
    )
    client = SimpleNamespace(responses=responses)
    rewrite = create_query_rewriter(client, model_name="test-model")

    result = rewrite(
        question="传音的竞争优势是否持续？",
        query="传音优势",
        evidence=[{"text": "多品牌策略"}],
    )

    assert result == "传音 非洲 本土化 渠道 竞争壁垒"
    assert responses.calls[0]["text_format"] is RewrittenQuery
    assert "不要回答问题" in responses.calls[0]["input"][0]["content"]
    assert "多品牌策略" in responses.calls[0]["input"][1]["content"]


def test_query_rewriter_rejects_missing_parsed_output():
    client = SimpleNamespace(responses=FakeResponses(None))
    rewrite = create_query_rewriter(client, model_name="test-model")

    with pytest.raises(LLMJsonParseError):
        rewrite(question="问题", query="查询", evidence=[])
