from copy import deepcopy

import pytest

from src.evidence_registry import EvidenceRegistry


def make_local_record(**overrides):
    record = {
        "doc_id": "report_001",
        "page_number": 3,
        "chunk_id": "report_001:p3:c0",
        "content_type": "text",
        "text": "本土化运营形成竞争优势",
        "embedding": [0.1, 0.2],
        "score": 0.9,
        "rank": 1,
    }
    record.update(overrides)
    return record


def test_registry_assigns_stable_id_and_removes_transient_fields():
    registry = EvidenceRegistry()
    original = make_local_record()
    original_copy = deepcopy(original)

    first = registry.register([original], source_kind="local")[0]
    second = registry.register(
        [make_local_record(score=0.3, rank=8)],
        source_kind="local",
    )[0]

    assert first["evidence_id"].startswith("E-local-")
    assert second["evidence_id"] == first["evidence_id"]
    assert "embedding" not in first
    assert "score" not in first
    assert "rank" not in first
    assert first["source_kind"] == "local"
    assert len(registry) == 1
    assert original == original_copy


def test_registry_returns_defensive_copies():
    registry = EvidenceRegistry()
    registered = registry.register(
        [make_local_record()],
        source_kind="local",
    )[0]

    fetched = registry.get(registered["evidence_id"])
    fetched["text"] = "被下游修改"

    assert registry.get(registered["evidence_id"])["text"] == (
        "本土化运营形成竞争优势"
    )


def test_registry_rejects_explicit_id_collision():
    registry = EvidenceRegistry()
    registry.register(
        [make_local_record(evidence_id="E1")],
        source_kind="local",
    )

    with pytest.raises(ValueError, match="不同内容"):
        registry.register(
            [
                make_local_record(
                    evidence_id="E1",
                    text="另一条证据",
                )
            ],
            source_kind="local",
        )


def test_registry_versions_changed_web_summary_without_overwriting():
    registry = EvidenceRegistry()
    first = registry.register(
        [
            {
                "url": "https://example.com/report",
                "text": "第一次搜索摘要",
            }
        ],
        source_kind="web",
    )[0]
    second = registry.register(
        [
            {
                "url": "https://example.com/report",
                "text": "另一次查询产生的不同摘要",
            }
        ],
        source_kind="web",
    )[0]

    assert second["evidence_id"] != first["evidence_id"]
    assert registry.get(first["evidence_id"])["text"] == "第一次搜索摘要"
    assert len(registry) == 2


def test_registry_get_many_rejects_unknown_id():
    registry = EvidenceRegistry()

    with pytest.raises(KeyError, match="未注册"):
        registry.get_many(["missing"])
