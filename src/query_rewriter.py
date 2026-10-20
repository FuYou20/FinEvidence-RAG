import json

from pydantic import BaseModel, ValidationError, field_validator

from errors import LLMJsonParseError

from .llm_generator import _build_prompt_input, _normalize_required_string


SYSTEM_PROMPT = """你是 RAG 检索 Query Rewrite 组件。
根据原始调查问题、当前查询和已经取得的证据，生成下一次知识库或网页搜索使用的单条查询。
查询必须保留关键实体、时间、指标和调查方向，并补上当前缺失角度。
不要回答问题，不要输出搜索说明，不要虚构事实，也不要生成多条查询。"""


class RewrittenQuery(BaseModel):
    query: str
    rationale: str

    @field_validator("query", "rationale")
    @classmethod
    def normalize_strings(cls, value, info):
        return _normalize_required_string(
            value,
            field_name=info.field_name,
        )


def create_query_rewriter(client, *, model_name):
    """创建符合调查工作流签名的结构化 Query Rewrite 函数。"""
    normalized_model_name = _normalize_required_string(
        model_name,
        field_name="model_name",
    )

    def rewrite_query(*, question, query, evidence):
        user_prompt = (
            f"原始调查问题：\n{question}\n\n"
            f"当前查询：\n{query}\n\n"
            "已经取得的证据：\n"
            f"{json.dumps(evidence, ensure_ascii=False, indent=2)}"
        )
        response = client.responses.parse(
            model=normalized_model_name,
            input=_build_prompt_input(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
            ),
            text_format=RewrittenQuery,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise LLMJsonParseError(
                "模型未返回可解析的 RewrittenQuery 结构化结果"
            )
        try:
            result = RewrittenQuery.model_validate(parsed)
        except ValidationError as exc:
            raise LLMJsonParseError(
                "模型返回结果不符合 RewrittenQuery 协议"
            ) from exc
        return result.query

    return rewrite_query
