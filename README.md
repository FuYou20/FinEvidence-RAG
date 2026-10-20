# FinEvidence-RAG

面向证券研报的证据驱动 Agentic RAG 系统。项目覆盖 PDF 正文与表格解析、混合召回、邻居 Chunk 扩展、重排、引用校验、主动拒答以及按证据缺口继续调查的完整链路，并通过 A2A 协议将调查能力封装为独立 Agent 服务。

## 项目概况

- 原始语料目录包含 281 份 PDF 研报；当前公开版本不包含研报原文件。
- 首版实验选取 12 份跨公司、跨报告类型的代表性研报，构建 1,201 个正文及表格 Chunk。
- 独立测试集 `Recall@5` 达到 90%，邻居扩展与重排使枚举项目召回率提升 18.9 个百分点。
- Redis 请求缓存与分布式锁可合并相同并发调查；实验中 8 路相同请求只执行 1 次真实调查。

## 核心链路

```text
PDF 正文/表格解析
        ↓
固定窗口切分与表格行切分
        ↓
Embedding + FAISS + BM25
        ↓
邻居 Chunk 扩展 + Rerank
        ↓
证据注册、引用与充分性判断
        ↓
Query Rewrite / 联网补证 / 主动拒答
```

调查循环采用“模型负责结构化决策，Python 负责执行”的方式。证据判断器输出 `finish`、`rewrite_query`、`multimodal` 或 `web_search`，应用层负责调用工具、合并证据、记录轨迹和限制最大步数。

## 工程特性

- 使用 PyMuPDF 提取正文，pdfplumber 提取有线表格；表格转换为 Markdown 后与正文统一入库。
- 为证据生成稳定 `evidence_id`，绑定文档、页码、Chunk 和原文；模型引用必须通过应用层合法性检查。
- 通过 Pydantic 约束证据判断、调查结果和带引用答案；证据不足时禁止输出事实性结论。
- 使用 Redis 缓存完整 `EvidenceBundle`，通过“缓存检查—获取锁—二次检查—回源调查”避免缓存击穿。
- 发布 Agent Card，通过 A2A Message、Task 和 Artifact 接收调查请求并返回结构化证据包。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

将自有 PDF 放入 `data/pdf/`，配置语料清单和模型密钥后构建索引。启动 A2A 服务：

```bash
python -m src.scripts.a2a_evidence_server
```

运行测试：

```bash
pytest -q
```

## 数据说明

证券研报 PDF、模型密钥、向量索引、Redis 数据和运行日志均未提交。仓库只保留实现代码、测试和配置模板；请使用具有合法授权的数据自行构建知识库。

## 当前边界

- 图片和扫描件的多模态解析接口已经预留，公开版本尚未接入视觉模型。
- A2A Task Store 当前为内存实现，服务重启后任务状态不会恢复。
- 检索和生成效果依赖用户提供的语料、Embedding、Rerank 与生成模型。
