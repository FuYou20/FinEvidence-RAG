from copy import deepcopy

import pytest

from src.evaluation.retrieval_dataset import validate_retrieval_dataset


def _metadata():
    return [
        {
            "chunk_id": "doc_dev:p1:c0",
            "doc_id": "doc_dev",
            "page_number": 1,
            "document_chunk_index": 0,
            "content_type": "text",
            "text": "第一段正文证据",
        },
        {
            "chunk_id": "doc_dev:p1:c1",
            "doc_id": "doc_dev",
            "page_number": 1,
            "document_chunk_index": 1,
            "content_type": "text",
            "text": "第二段正文证据",
        },
        {
            "chunk_id": "doc_test:p2:t0:c0",
            "doc_id": "doc_test",
            "page_number": 2,
            "document_chunk_index": 0,
            "content_type": "table",
            "text": "| 指标 | 数值 |",
        },
    ]


def _evidence(source):
    return {
        "doc_id": source["doc_id"],
        "page_number": source["page_number"],
        "relevance": 3,
        "evidence_role": "primary",
        "evidence_type": source["content_type"],
        "source_chunk_id": source["chunk_id"],
        "quote": source["text"],
    }


def test_validate_retrieval_dataset_checks_traceability_and_split():
    metadata = _metadata()
    cases = [
        {
            "query_id": "q1",
            "question": "正文问题",
            "category": "boundary",
            "split": "dev",
            "gold_evidence": [_evidence(metadata[0]), _evidence(metadata[1])],
        },
        {
            "query_id": "q2",
            "question": "表格问题",
            "category": "table",
            "split": "test",
            "gold_evidence": [_evidence(metadata[2])],
        },
    ]

    summary = validate_retrieval_dataset(
        cases,
        metadata,
        expected_total=2,
        require_disjoint_splits=True,
    )

    assert summary["split_counts"] == {"dev": 1, "test": 1}
    assert summary["split_document_overlap"] == []


def test_validate_retrieval_dataset_rejects_modified_quote():
    metadata = _metadata()
    case = {
        "query_id": "q1",
        "question": "正文问题",
        "category": "direct_text",
        "split": "dev",
        "gold_evidence": [_evidence(metadata[0])],
    }
    case["gold_evidence"][0]["quote"] = "模型改写后的伪引文"

    with pytest.raises(ValueError, match="quote.*metadata 不一致"):
        validate_retrieval_dataset([case], metadata)


def test_validate_retrieval_dataset_rejects_non_adjacent_boundary():
    metadata = _metadata()
    far_source = deepcopy(metadata[1])
    far_source["chunk_id"] = "doc_dev:p3:c0"
    far_source["page_number"] = 3
    far_source["document_chunk_index"] = 8
    metadata.append(far_source)
    case = {
        "query_id": "q1",
        "question": "跨边界问题",
        "category": "boundary",
        "split": "dev",
        "gold_evidence": [_evidence(metadata[0]), _evidence(far_source)],
    }

    with pytest.raises(ValueError, match="并不相邻"):
        validate_retrieval_dataset([case], metadata)
