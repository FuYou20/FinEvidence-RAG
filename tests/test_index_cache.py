from copy import deepcopy
import hashlib
import json

import faiss
import numpy as np
import pytest

from src.index_cache import (
    compute_corpus_fingerprint,
    compute_file_sha256,
    load_index_bundle,
    load_or_build_index,
    save_index_bundle,
)


def make_saved_bundle(tmp_path):
    cache_dir = tmp_path / "cache"
    index = faiss.IndexFlatIP(2)
    index.add(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    metadata = [
        {"chunk_id": "doc:p1:c0", "text": "第一条"},
        {"chunk_id": "doc:p2:c0", "text": "第二条"},
    ]
    build_config = {
        "schema_version": 1,
        "embedding_model": "fake-model",
        "chunk_size": 512,
        "overlap": 64,
    }
    save_index_bundle(cache_dir, index, metadata, build_config)
    return cache_dir, metadata, build_config


def test_compute_file_sha256_reads_all_blocks_and_matches_expected(tmp_path):
    content = b"a" * (1024 * 1024 + 37)
    source = tmp_path / "large.bin"
    source.write_bytes(content)

    result = compute_file_sha256(source)

    assert result == hashlib.sha256(content).hexdigest()


def test_compute_file_sha256_accepts_path_and_string(tmp_path):
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"same source")

    assert compute_file_sha256(source) == compute_file_sha256(str(source))


def test_compute_file_sha256_propagates_missing_file_error(tmp_path):
    missing = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        compute_file_sha256(missing)


def test_compute_file_sha256_rejects_unsupported_path_type():
    with pytest.raises(TypeError, match="str|Path"):
        compute_file_sha256(123)


def test_save_index_bundle_writes_reloadable_files_without_mutating_config(tmp_path):
    index = faiss.IndexFlatIP(3)
    index.add(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    metadata = [
        {"chunk_id": "doc:p1:c0", "text": "第一条"},
        {"chunk_id": "doc:p2:c0", "text": "第二条"},
    ]
    build_config = {
        "schema_version": 1,
        "embedding_model": "fake-model",
        "chunk_size": 512,
    }
    original_config = deepcopy(build_config)
    cache_dir = tmp_path / "nested" / "cache"

    save_index_bundle(cache_dir, index, metadata, build_config)

    restored_index = faiss.read_index(str(cache_dir / "index.faiss"))
    assert restored_index.ntotal == 2
    assert restored_index.d == 3

    with (cache_dir / "metadata.jsonl").open(encoding="utf-8") as file:
        restored_metadata = [json.loads(line) for line in file]
    assert restored_metadata == metadata

    saved_config = json.loads(
        (cache_dir / "build_config.json").read_text(encoding="utf-8")
    )
    assert saved_config == {
        **build_config,
        "index_size": 2,
        "index_dimension": 3,
    }
    assert build_config == original_config


def test_save_index_bundle_rejects_index_metadata_count_mismatch(tmp_path):
    index = faiss.IndexFlatIP(2)
    index.add(np.array([[1.0, 0.0]], dtype=np.float32))

    with pytest.raises(ValueError, match="数量不匹配"):
        save_index_bundle(tmp_path / "cache", index, [], {})


@pytest.mark.parametrize(
    ("metadata", "build_config", "expected_message"),
    [
        ({}, {}, "metadata"),
        ([], [], "build_config"),
    ],
)
def test_save_index_bundle_rejects_invalid_container_types(
    tmp_path,
    metadata,
    build_config,
    expected_message,
):
    index = faiss.IndexFlatIP(2)

    with pytest.raises(TypeError, match=expected_message):
        save_index_bundle(tmp_path / "cache", index, metadata, build_config)


def test_load_index_bundle_round_trips_saved_index_metadata_and_config(tmp_path):
    cache_dir, expected_metadata, original_config = make_saved_bundle(tmp_path)

    index, metadata, build_config = load_index_bundle(
        cache_dir,
        expected_config={
            "embedding_model": "fake-model",
            "chunk_size": 512,
            "overlap": 64,
        },
    )

    assert index.ntotal == 2
    assert index.d == 2
    assert metadata == expected_metadata
    assert build_config == {
        **original_config,
        "index_size": 2,
        "index_dimension": 2,
    }


@pytest.mark.parametrize(
    "missing_name",
    ["index.faiss", "metadata.jsonl", "build_config.json"],
)
def test_load_index_bundle_reports_each_missing_file(tmp_path, missing_name):
    cache_dir, _metadata, _config = make_saved_bundle(tmp_path)
    (cache_dir / missing_name).unlink()

    with pytest.raises(FileNotFoundError, match=missing_name):
        load_index_bundle(cache_dir)


@pytest.mark.parametrize(
    "expected_config",
    [
        {"embedding_model": "different-model"},
        {"unknown_setting": True},
    ],
)
def test_load_index_bundle_rejects_incompatible_expected_config(
    tmp_path,
    expected_config,
):
    cache_dir, _metadata, _config = make_saved_bundle(tmp_path)

    with pytest.raises(ValueError):
        load_index_bundle(cache_dir, expected_config=expected_config)


def test_load_index_bundle_rejects_non_dict_expected_config(tmp_path):
    cache_dir, _metadata, _config = make_saved_bundle(tmp_path)

    with pytest.raises(TypeError, match="expected_config"):
        load_index_bundle(cache_dir, expected_config=["not", "a", "dict"])


def test_load_index_bundle_rejects_metadata_count_mismatch(tmp_path):
    cache_dir, _metadata, _config = make_saved_bundle(tmp_path)
    (cache_dir / "metadata.jsonl").write_text(
        json.dumps({"chunk_id": "only-one", "text": "残缺"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"len\(metadata\)"):
        load_index_bundle(cache_dir)


def test_load_index_bundle_rejects_missing_integrity_field(tmp_path):
    cache_dir, _metadata, _config = make_saved_bundle(tmp_path)
    config_path = cache_dir / "build_config.json"
    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    saved_config.pop("index_dimension")
    config_path.write_text(
        json.dumps(saved_config, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="index_dimension"):
        load_index_bundle(cache_dir)


def make_builder_result():
    index = faiss.IndexFlatIP(2)
    index.add(np.array([[1.0, 0.0]], dtype=np.float32))
    metadata = [{"chunk_id": "rebuilt:p1:c0", "text": "新索引"}]
    return index, metadata


def test_load_or_build_index_uses_valid_cache_without_calling_builder(tmp_path):
    cache_dir, expected_metadata, build_config = make_saved_bundle(tmp_path)

    def builder_must_not_run():
        raise AssertionError("有效缓存命中时不应调用 builder")

    index, metadata, cache_status = load_or_build_index(
        cache_dir,
        builder_must_not_run,
        build_config=build_config,
    )

    assert index.ntotal == 2
    assert metadata == expected_metadata
    assert cache_status == "hit"


def test_load_or_build_index_builds_once_then_reuses_saved_cache(tmp_path):
    cache_dir = tmp_path / "missing-cache"
    build_config = {
        "schema_version": 1,
        "embedding_model": "fake-model",
        "chunk_size": 512,
    }
    call_count = 0

    def builder():
        nonlocal call_count
        call_count += 1
        return make_builder_result()

    first_index, first_metadata, first_cache_status = load_or_build_index(
        cache_dir,
        builder,
        build_config=build_config,
    )
    second_index, second_metadata, second_cache_status = load_or_build_index(
        cache_dir,
        builder,
        build_config=build_config,
    )

    assert call_count == 1
    assert first_cache_status == "rebuilt"
    assert second_cache_status == "hit"
    assert first_index.ntotal == second_index.ntotal == 1
    assert first_metadata == second_metadata
    assert (cache_dir / "index.faiss").is_file()
    assert (cache_dir / "metadata.jsonl").is_file()
    assert (cache_dir / "build_config.json").is_file()


def test_load_or_build_index_rebuilds_when_config_changes(tmp_path):
    cache_dir, _metadata, old_config = make_saved_bundle(tmp_path)
    new_config = {**old_config, "chunk_size": 256}
    call_count = 0

    def builder():
        nonlocal call_count
        call_count += 1
        return make_builder_result()

    index, metadata, cache_status = load_or_build_index(
        cache_dir,
        builder,
        build_config=new_config,
    )

    assert call_count == 1
    assert cache_status == "rebuilt"
    assert index.ntotal == 1
    assert metadata[0]["chunk_id"] == "rebuilt:p1:c0"
    _loaded_index, _loaded_metadata, saved_config = load_index_bundle(cache_dir)
    assert saved_config["chunk_size"] == 256


def test_load_or_build_index_rebuilds_corrupt_metadata(tmp_path):
    cache_dir, _metadata, build_config = make_saved_bundle(tmp_path)
    (cache_dir / "metadata.jsonl").write_text(
        json.dumps({"chunk_id": "only-one", "text": "残缺"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    call_count = 0

    def builder():
        nonlocal call_count
        call_count += 1
        return make_builder_result()

    index, metadata, cache_status = load_or_build_index(
        cache_dir,
        builder,
        build_config=build_config,
    )

    assert call_count == 1
    assert cache_status == "rebuilt"
    assert index.ntotal == len(metadata) == 1


def test_load_or_build_index_propagates_builder_failure(tmp_path):
    def failing_builder():
        raise RuntimeError("embedding service unavailable")

    with pytest.raises(RuntimeError, match="embedding service unavailable"):
        load_or_build_index(
            tmp_path / "missing-cache",
            failing_builder,
            build_config={"schema_version": 1},
        )


@pytest.mark.parametrize(
    ("builder", "build_config", "error_type"),
    [
        (None, {"schema_version": 1}, TypeError),
        (make_builder_result, [], TypeError),
        (make_builder_result, {}, ValueError),
    ],
)
def test_load_or_build_index_validates_arguments(
    tmp_path,
    builder,
    build_config,
    error_type,
):
    with pytest.raises(error_type):
        load_or_build_index(
            tmp_path / "cache",
            builder,
            build_config=build_config,
        )


def make_corpus_files(tmp_path):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "a.pdf").write_bytes(b"pdf-a-v1")
    (tmp_path / "reports" / "b.pdf").write_bytes(b"pdf-b-v1")
    records = [
        {"doc_id": "doc-a", "relative_path": "reports/a.pdf"},
        {"doc_id": "doc-b", "relative_path": "reports/b.pdf"},
    ]
    return records


def test_compute_corpus_fingerprint_is_stable_and_order_independent(tmp_path):
    records = make_corpus_files(tmp_path)

    first = compute_corpus_fingerprint(records, str(tmp_path))
    second = compute_corpus_fingerprint(list(reversed(records)), tmp_path)

    assert first == second
    assert len(first) == 64


def test_compute_corpus_fingerprint_changes_when_pdf_content_changes(tmp_path):
    records = make_corpus_files(tmp_path)
    before = compute_corpus_fingerprint(records, tmp_path)

    (tmp_path / "reports" / "a.pdf").write_bytes(b"pdf-a-v2")
    after = compute_corpus_fingerprint(records, tmp_path)

    assert after != before


def test_compute_corpus_fingerprint_changes_when_document_is_added_or_removed(
    tmp_path,
):
    records = make_corpus_files(tmp_path)
    full = compute_corpus_fingerprint(records, tmp_path)
    removed = compute_corpus_fingerprint(records[:1], tmp_path)

    (tmp_path / "reports" / "c.pdf").write_bytes(b"pdf-c")
    added = compute_corpus_fingerprint(
        records
        + [{"doc_id": "doc-c", "relative_path": "reports/c.pdf"}],
        tmp_path,
    )

    assert removed != full
    assert added != full
    assert added != removed


@pytest.mark.parametrize(
    ("records", "error_type"),
    [
        (None, TypeError),
        ({}, TypeError),
        ([], ValueError),
    ],
)
def test_compute_corpus_fingerprint_validates_records_type_and_content(
    tmp_path,
    records,
    error_type,
):
    with pytest.raises(error_type):
        compute_corpus_fingerprint(records, tmp_path)


def test_compute_corpus_fingerprint_rejects_invalid_project_root(tmp_path):
    with pytest.raises(ValueError, match="project_root"):
        compute_corpus_fingerprint(
            [{"doc_id": "doc", "relative_path": "report.pdf"}],
            tmp_path / "missing-root",
        )


def test_compute_corpus_fingerprint_rejects_path_traversal(tmp_path):
    outside = tmp_path.parent / "outside.pdf"
    outside.write_bytes(b"outside")

    with pytest.raises(ValueError, match="越过"):
        compute_corpus_fingerprint(
            [{"doc_id": "outside", "relative_path": "../outside.pdf"}],
            tmp_path,
        )


def test_compute_corpus_fingerprint_reports_missing_pdf(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.pdf"):
        compute_corpus_fingerprint(
            [{"doc_id": "missing", "relative_path": "missing.pdf"}],
            tmp_path,
        )


def test_compute_corpus_fingerprint_rejects_directory_as_pdf(tmp_path):
    (tmp_path / "folder.pdf").mkdir()

    with pytest.raises(ValueError, match="不是文件"):
        compute_corpus_fingerprint(
            [{"doc_id": "folder", "relative_path": "folder.pdf"}],
            tmp_path,
        )
