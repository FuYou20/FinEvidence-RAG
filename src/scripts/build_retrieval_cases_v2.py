"""从已审核语料中生成并验证检索评测集 v2。

模型只负责提出问题和答案；证据组合及原文引用均由程序从索引 metadata
原样回填。输出仍需要最终人工语义复核。
"""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re

from dotenv import load_dotenv
from openai import OpenAI

from src.corpus_indexer import load_corpus_manifest


CATEGORIES = (
    "direct_text",
    "boundary",
    "table",
    "text_table",
    "multi_section",
)
CATEGORY_EXCLUDED_DOC_INDICES = {
    "direct_text": {10, 11},
    "boundary": {8, 9},
    "table": {6, 7},
    "text_table": {4, 5},
    "multi_section": {2, 3},
}
# 这7份文档一共产生30条开发案例，且每类恰好6条；其余为锁定测试集。
DEV_DOC_INDICES = {0, 1, 2, 4, 6, 8, 10}


def parse_args():
    parser = argparse.ArgumentParser(description="生成检索评测集 v2 草案")
    parser.add_argument(
        "--output",
        default="data/eval/retrieval_cases_v2.jsonl",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="模型名称；Qwen 默认使用 qwen-flash",
    )
    parser.add_argument(
        "--provider",
        choices=("qwen", "deepseek"),
        default="qwen",
    )
    parser.add_argument("--max-attempts", type=int, default=4)
    return parser.parse_args()


def _evenly_select(items, limit):
    if len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[len(items) // 2]]
    positions = {
        round(index * (len(items) - 1) / (limit - 1))
        for index in range(limit)
    }
    return [items[position] for position in sorted(positions)]


def _normalize_whitespace(text):
    return " ".join(str(text).split())


def _is_useful_text(item):
    text = _normalize_whitespace(item.get("text", ""))
    if len(text) < 160:
        return False
    noisy_markers = ("正文目录", "免责声明", "证券研究报告 请阅读")
    return not any(marker in text[:180] for marker in noisy_markers)


def _catalog_record(item):
    return {
        "source_id": item["chunk_id"],
        "page_number": item["page_number"],
        "document_chunk_index": item.get("document_chunk_index"),
        "content_type": item.get("content_type", "text"),
        "table_id": item.get("table_id"),
        "table_caption": item.get("table_caption"),
        "text": item["text"],
    }


def _pick_at_fraction(items, fraction):
    if not items:
        raise ValueError("候选列表不能为空")
    position = min(round((len(items) - 1) * fraction), len(items) - 1)
    return items[position]


def build_document_catalog(document_rows, categories, *, doc_index):
    rows = sorted(
        document_rows,
        key=lambda item: item.get("document_chunk_index", 0),
    )
    text_rows = [item for item in rows if item.get("content_type") == "text"]
    useful_text_rows = [item for item in text_rows if _is_useful_text(item)]
    table_rows = [
        item
        for item in rows
        if item.get("content_type") == "table"
        and len(_normalize_whitespace(item.get("text", ""))) >= 80
    ]

    boundary_pairs = []
    for left, right in zip(text_rows, text_rows[1:]):
        left_index = left.get("document_chunk_index")
        right_index = right.get("document_chunk_index")
        if (
            isinstance(left_index, int)
            and isinstance(right_index, int)
            and right_index == left_index + 1
            and _is_useful_text(left)
            and _is_useful_text(right)
        ):
            boundary_pairs.append(
                {
                    "left_source_id": left["chunk_id"],
                    "right_source_id": right["chunk_id"],
                    "left_page": left["page_number"],
                    "right_page": right["page_number"],
                }
            )

    if not useful_text_rows:
        raise ValueError("文档没有可用正文 Chunk")
    if any(category in categories for category in ("table", "text_table")):
        if not table_rows:
            raise ValueError("文档没有可用表格 Chunk")
    if "boundary" in categories and not boundary_pairs:
        raise ValueError("文档没有可用相邻正文 Chunk")

    # 先由程序确定每个类别的证据组合。模型只负责把指定证据改写成问题，
    # 避免让模型在几十个候选中同时承担检索和标注两个职责。
    shift = (doc_index % 5) * 0.06
    category_sources = {}
    if "direct_text" in categories:
        direct = _pick_at_fraction(useful_text_rows, 0.18 + shift)
        category_sources["direct_text"] = [direct["chunk_id"]]
    if "boundary" in categories:
        boundary = _pick_at_fraction(boundary_pairs, 0.35 + shift)
        category_sources["boundary"] = [
            boundary["left_source_id"],
            boundary["right_source_id"],
        ]
    if "table" in categories:
        table = _pick_at_fraction(table_rows, 0.42 + shift)
        category_sources["table"] = [table["chunk_id"]]
    if "text_table" in categories:
        table = _pick_at_fraction(table_rows, 0.68 + shift)
        table_index = table.get("document_chunk_index", 0)
        nearest_text = min(
            useful_text_rows,
            key=lambda item: abs(
                item.get("document_chunk_index", 0) - table_index
            ),
        )
        category_sources["text_table"] = [
            nearest_text["chunk_id"],
            table["chunk_id"],
        ]
    if "multi_section" in categories:
        left = _pick_at_fraction(useful_text_rows, 0.30 + shift / 2)
        distinct_page_rows = [
            item
            for item in useful_text_rows
            if item["page_number"] != left["page_number"]
        ]
        right = _pick_at_fraction(distinct_page_rows, 0.72 - shift / 2)
        category_sources["multi_section"] = [
            left["chunk_id"],
            right["chunk_id"],
        ]

    source_ids = {
        source_id
        for assigned_sources in category_sources.values()
        for source_id in assigned_sources
    }
    sources = [
        _catalog_record(item)
        for item in rows
        if item["chunk_id"] in source_ids
    ]
    return {
        "sources": sources,
        "category_sources": category_sources,
    }


def _extract_json_object(text):
    normalized = text.strip()
    normalized = re.sub(r"^```(?:json)?\s*", "", normalized)
    normalized = re.sub(r"\s*```$", "", normalized)
    start = normalized.find("{")
    end = normalized.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("模型输出中没有 JSON 对象")
    # 部分兼容模型会在 JSON 字符串中输出未经转义的换行符；strict=False
    # 只放宽控制字符解析，后续字段和证据协议仍由显式校验负责。
    return json.loads(normalized[start : end + 1], strict=False)


def _build_prompt(record, categories, catalog, validation_feedback=None):
    feedback = (
        f"\n上一次生成未通过验证：{validation_feedback}\n请重新生成全部案例。"
        if validation_feedback
        else ""
    )
    return f"""
你正在为金融研报 RAG 建立人工评测集草案。

文档信息：
- doc_id: {record['doc_id']}
- 公司: {record['company']}
- 标题: {record['title']}
- 需要生成的类别: {json.dumps(categories, ensure_ascii=False)}

类别规则：
1. direct_text：使用至少一条正文证据，问题询问明确事实。
2. boundary：答案需要两个相邻正文 Chunk 才完整。
3. table：必须使用表格证据，询问明确数字、占比、时间或表内比较。
4. text_table：必须同时使用一条正文和一条表格证据，两者缺一不可。
5. multi_section：至少使用不同页的两条正文证据，回答需要归纳多个方面。

统一要求：
- 每个指定类别恰好生成1条，不能生成其他类别。
- question 必须包含完整公司名“{record['company']}”，不能只写“公司”。
- 每个类别必须使用 catalog.category_sources 为该类别指定的全部 source_id，不能增删或替换。
- quote 字段可直接复制对应 source 的 text；最终引文会由程序从 metadata 原样回填。
- relevance 使用1、2或3，核心证据为3，辅助证据为1或2。
- evidence_role 只能是 primary 或 supporting。
- reference_answer 必须完全由所列证据支持，不得使用外部知识。
- 不要输出 Markdown，只输出合法 JSON。

JSON格式：
{{
  "cases": [
    {{
      "category": "direct_text",
      "question": "...",
      "difficulty": "easy|medium|hard",
      "gold_evidence": [
        {{
          "source_id": "...",
          "quote": "...",
          "relevance": 3,
          "evidence_role": "primary"
        }}
      ],
      "reference_answer": "..."
    }}
  ]
}}

候选证据 catalog：
{json.dumps(catalog, ensure_ascii=False)}
{feedback}
""".strip()


def validate_and_normalize_document_cases(
    payload,
    *,
    record,
    categories,
    source_by_id,
    split,
    category_sources,
):
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError("顶层必须包含 cases 列表")
    cases = payload["cases"]
    if len(cases) != len(categories):
        raise ValueError(
            f"需要 {len(categories)} 条案例，实际得到 {len(cases)} 条"
        )
    actual_categories = [case.get("category") for case in cases]
    if Counter(actual_categories) != Counter(categories):
        raise ValueError(
            f"类别不匹配: expected={categories}, actual={actual_categories}"
        )

    normalized_cases = []
    for case in cases:
        category = case["category"]
        question = case.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{category} 的 question 必须是非空字符串")
        question = question.strip()
        if record["company"] not in question:
            raise ValueError(f"{category} 的问题必须包含公司名")
        difficulty = case.get("difficulty")
        if difficulty not in {"easy", "medium", "hard"}:
            raise ValueError(f"{category} 的 difficulty 无效")
        reference_answer = case.get("reference_answer")
        if not isinstance(reference_answer, str) or not reference_answer.strip():
            raise ValueError(f"{category} 的 reference_answer 不能为空")
        raw_evidence = case.get("gold_evidence")
        if not isinstance(raw_evidence, list) or not raw_evidence:
            raise ValueError(f"{category} 必须包含 gold_evidence")

        evidence = []
        seen_source_ids = set()
        for item in raw_evidence:
            if not isinstance(item, dict):
                raise ValueError(f"{category} 的 evidence 必须是对象")
            source_id = item.get("source_id")
            if source_id not in source_by_id:
                raise ValueError(f"{category} 使用未知 source_id: {source_id}")
            if source_id in seen_source_ids:
                raise ValueError(f"{category} 重复使用 source_id: {source_id}")
            source = source_by_id[source_id]
            # 模型只声明 source_id，最终 Gold 引文始终从已索引的原文回填，
            # 避免模型在抄写过程中润色、删字或改变标点。
            quote = source["text"].strip()
            relevance = item.get("relevance")
            if relevance not in {1, 2, 3}:
                raise ValueError(f"{category} 的 relevance 必须是1、2或3")
            evidence_role = item.get("evidence_role")
            if evidence_role not in {"primary", "supporting"}:
                raise ValueError(f"{category} 的 evidence_role 无效")
            normalized_evidence = {
                "doc_id": record["doc_id"],
                "page_number": source["page_number"],
                "relevance": relevance,
                "evidence_role": evidence_role,
                "evidence_type": source.get("content_type", "text"),
                "source_chunk_id": source_id,
                "quote": quote.strip(),
            }
            if source.get("table_id"):
                normalized_evidence["table_id"] = source["table_id"]
            evidence.append(normalized_evidence)
            seen_source_ids.add(source_id)

        expected_source_ids = set(category_sources[category])
        if seen_source_ids != expected_source_ids:
            raise ValueError(
                f"{category} 必须使用指定证据: "
                f"expected={sorted(expected_source_ids)}, "
                f"actual={sorted(seen_source_ids)}"
            )

        evidence_types = {item["evidence_type"] for item in evidence}
        if category == "direct_text" and "text" not in evidence_types:
            raise ValueError("direct_text 必须包含正文证据")
        if category == "table" and "table" not in evidence_types:
            raise ValueError("table 必须包含表格证据")
        if category == "text_table" and evidence_types != {"text", "table"}:
            raise ValueError("text_table 必须同时包含正文和表格证据")
        if category == "multi_section":
            if evidence_types != {"text"} or len(
                {item["page_number"] for item in evidence}
            ) < 2:
                raise ValueError("multi_section 必须使用不同页的正文证据")
        if category == "boundary":
            if len(evidence) != 2 or evidence_types != {"text"}:
                raise ValueError("boundary 必须恰好使用两条正文证据")
            source_indices = sorted(
                source_by_id[item["source_chunk_id"]]["document_chunk_index"]
                for item in evidence
            )
            if source_indices[1] != source_indices[0] + 1:
                raise ValueError("boundary 的两条证据不是相邻 Chunk")

        normalized_cases.append(
            {
                "query_id": None,
                "question": question,
                "category": category,
                "difficulty": difficulty,
                "split": split,
                "annotation_status": "evidence_verified",
                "gold_evidence": evidence,
                "reference_answer": reference_answer.strip(),
            }
        )
    return normalized_cases


def validate_dataset(cases):
    if len(cases) != 50:
        raise ValueError(f"评测集必须有50条，实际为 {len(cases)}")
    category_counts = Counter(case["category"] for case in cases)
    if category_counts != Counter({category: 10 for category in CATEGORIES}):
        raise ValueError(f"类别数量不平衡: {dict(category_counts)}")
    split_counts = Counter(case["split"] for case in cases)
    if split_counts != Counter({"dev": 30, "test": 20}):
        raise ValueError(f"数据集切分数量错误: {dict(split_counts)}")
    for split in ("dev", "test"):
        counts = Counter(
            case["category"] for case in cases if case["split"] == split
        )
        expected = 6 if split == "dev" else 4
        if counts != Counter({category: expected for category in CATEGORIES}):
            raise ValueError(f"{split} 类别不平衡: {dict(counts)}")
    questions = [case["question"] for case in cases]
    if len(set(questions)) != len(questions):
        raise ValueError("评测集中存在重复问题")


def main():
    args = parse_args()
    if args.max_attempts <= 0:
        raise ValueError("max_attempts 必须是正整数")
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    if args.provider == "qwen":
        api_key = os.getenv("DASHSCOPE_API_KEY")
        base_url = os.getenv("QWEN_EMBEDDING_BASE_URL")
        model_name = args.model or "qwen-flash"
    else:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL")
        model_name = args.model or os.getenv("DEEPSEEK_MODEL")
    if not api_key or not base_url or not model_name:
        raise ValueError(f"缺少 {args.provider} API 配置")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=90.0,
        max_retries=0,
    )
    records = load_corpus_manifest(project_root / "data/eval/corpus_v1.jsonl")
    metadata_path = project_root / "data/cache/retrieval_index/metadata.jsonl"
    metadata = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows_by_doc = defaultdict(list)
    for item in metadata:
        rows_by_doc[item["doc_id"]].append(item)
    source_by_id = {item["chunk_id"]: item for item in metadata}

    output_path = project_root / args.output
    checkpoint_path = output_path.with_suffix(".checkpoint.json")
    checkpoint = {}
    if checkpoint_path.is_file():
        raw_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if isinstance(raw_checkpoint, dict):
            checkpoint = raw_checkpoint

    all_cases = []
    system_prompt = (
        "你是严谨的金融研报检索评测集标注员。你只能使用用户提供的"
        "候选证据，不得补充常识，不得改写引文，并严格返回JSON。"
    )
    for doc_index, record in enumerate(records):
        categories = [
            category
            for category in CATEGORIES
            if doc_index not in CATEGORY_EXCLUDED_DOC_INDICES[category]
        ]
        split = "dev" if doc_index in DEV_DOC_INDICES else "test"
        if record["doc_id"] in checkpoint:
            resumed_cases = checkpoint[record["doc_id"]]
            all_cases.extend(resumed_cases)
            print(
                f"[{record['doc_id']}] resumed {len(resumed_cases)} cases "
                f"from checkpoint"
            )
            continue
        catalog = build_document_catalog(
            rows_by_doc[record["doc_id"]],
            categories,
            doc_index=doc_index,
        )
        feedback = None
        for attempt in range(1, args.max_attempts + 1):
            prompt = _build_prompt(
                record,
                categories,
                catalog,
                validation_feedback=feedback,
            )
            messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
            if args.provider == "qwen":
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                )
                output_text = response.choices[0].message.content
            else:
                response = client.responses.create(
                    model=model_name,
                    input=messages,
                )
                output_text = response.output_text
            try:
                payload = _extract_json_object(output_text)
                document_cases = validate_and_normalize_document_cases(
                    payload,
                    record=record,
                    categories=categories,
                    source_by_id=source_by_id,
                    split=split,
                    category_sources=catalog["category_sources"],
                )
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                feedback = str(exc)
                print(
                    f"[{record['doc_id']}] attempt {attempt} failed: {feedback}"
                )
                continue
            all_cases.extend(document_cases)
            checkpoint[record["doc_id"]] = document_cases
            checkpoint_path.write_text(
                json.dumps(checkpoint, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"[{record['doc_id']}] generated {len(document_cases)} cases "
                f"for {split}"
            )
            break
        else:
            raise RuntimeError(
                f"文档 {record['doc_id']} 在 {args.max_attempts} 次后仍未通过验证"
            )

    for index, case in enumerate(all_cases, start=1):
        case["query_id"] = f"ret_v2_{index:03d}"
    validate_dataset(all_cases)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for case in all_cases:
            file.write(json.dumps(case, ensure_ascii=False) + "\n")
    checkpoint_path.unlink(missing_ok=True)

    print(f"wrote {len(all_cases)} cases to {output_path}")
    print("category counts:", dict(Counter(c["category"] for c in all_cases)))
    print("split counts:", dict(Counter(c["split"] for c in all_cases)))


if __name__ == "__main__":
    main()
