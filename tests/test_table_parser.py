from src.table_parser import (
    find_table_caption,
    is_meaningful_table,
    normalize_rows,
    table_to_markdown,
)


def test_normalize_rows_cleans_cells_and_pads_short_rows():
    rows = normalize_rows([[" A\nB ", None], ["C"]])

    assert rows == [["A B", ""], ["C", ""]]


def test_meaningful_table_filter_rejects_sparse_layout_noise():
    assert not is_meaningful_table(
        [["", "", ""], ["", "27", ""], ["", "", ""]]
    )
    assert is_meaningful_table(
        [["年份", "营收"], ["2021", "100"]]
    )


def test_meaningful_table_filter_rejects_single_populated_column():
    assert not is_meaningful_table(
        [
            ["", "第一行很长的正文"],
            ["", "第二行很长的正文"],
            ["", "第三行很长的正文"],
        ]
    )


def test_table_to_markdown_escapes_pipe_characters():
    markdown = table_to_markdown(
        [["字段", "值"], ["A|B", "100"]]
    )

    assert "A\\|B" in markdown
    assert "| --- | --- |" in markdown


def test_find_table_caption_uses_closest_overlapping_title():
    lines = [
        {"text": "普通正文", "bbox": [10, 20, 200, 30]},
        {"text": "图表59：APP排行榜", "bbox": [30, 90, 300, 105]},
    ]

    caption = find_table_caption(lines, [20, 110, 500, 300])

    assert caption == "图表59：APP排行榜"
