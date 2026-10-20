# 检索评测集 v2

`data/eval/retrieval_cases_v2.jsonl` 是离线知识库检索阶段的固定评测集，
用于比较 Dense、BM25、Hybrid、邻居扩展和 Cross-Encoder 重排策略。

## 数据规模与切分

- 共 50 个问题，每类 10 个：`direct_text`、`boundary`、`table`、
  `text_table`、`multi_section`。
- `dev` 30 条，用于调参与消融；`test` 20 条，只用于最终报告。
- 按文档划分而不是按问题随机划分，`dev` 与 `test` 没有文档重叠，避免同一研报内容泄漏。

## 证据标注

每条 `gold_evidence` 都包含 `source_chunk_id`、页码、内容类型、相关性等级和
原文引文。引文由程序从索引 `metadata.jsonl` 原样复制，不能由模型改写。

- `relevance=3`：回答不可缺少的核心证据。
- `relevance=2`：能明显补全答案的支持证据。
- `relevance=1`：仅提供背景或弱支持。

`annotation_status=human_reviewed` 表示问题、答案和证据组合经过人工语义复核；
它不等于事实已由研报之外的独立来源核验。

## 类别含义

- `direct_text`：一段正文即可回答。
- `boundary`：答案跨越两个相邻正文 Chunk，测试固定长度切分造成的边界问题。
- `table`：核心答案来自表格。
- `text_table`：正文和表格缺一不可。
- `multi_section`：需要综合同一文档不同页面的正文证据。

## 验证与实验纪律

`validate_retrieval_dataset()` 会检查：案例数、类别分布、证据是否能回溯到索引、
引文是否逐字一致、类别约束，以及 `dev/test` 是否存在文档泄漏。

调参只看 `dev`；确定 Dense/Sparse 权重、邻居窗口和 rerank 候选数后，冻结配置，
再对 `test` 运行一次。主要报告 `Recall@K` 和 `nDCG@K`，同时保留
`Success@K`、MRR、平均/P95 延迟与失败案例。

建议的最小消融矩阵：

1. Dense baseline；
2. Hybrid；
3. Hybrid + 邻居扩展；
4. Hybrid + Cross-Encoder rerank；
5. Hybrid + 邻居扩展 + Cross-Encoder rerank。
