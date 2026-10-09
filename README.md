# RAG —— 学习项目

> **这个仓库用来学习 RAG，不是交付产品。**
> 代码的首要目标是能看懂、能跑通，其次才是抽象。每个结论都要求有实测支撑。

学习材料在 `D:\desktop\quackDocs\my_notes\知识库\raw\personal-notes\编程\RAG`，
一边读一边把结论落成代码和文档。

## 这是什么 / 不是什么

| 是 | 不是 |
|---|---|
| 学习过程的产物 | 生产可用的库 |
| 单体实现，允许内部不一致 | 已定型的对外接口 |
| 每个设计决定都有对应的探针实测 | 靠推断得出的结论 |
| 探针（`explore/`）与正式实现（`simple_rag/`）分开 | 两者混在一起 |

## 演进路线

**当前**：在 `simple_rag/` 里直接写实现。优先跑通和看懂，不追求抽象。

**之后**：进展到一定程度后，会把它**重写并抽象为一个独立的 SDK 包**，再引入到桌面项目
**LifePrism** 中使用。

顺序是有意的：先有能跑的具体实现，再从真实需求里提炼接口 —— 而不是一开始就为了"通用"
设计出用不上的抽象。

## 目录导览

| 目录 | 说明 |
|---|---|
| `simple_rag/` | 正式实现。9 个子包：`chunking` / `tokenization` / `db` / `embedding_api` / `rerank_api` / `repository` / `indexing` / `retrieval` / `config` |
| `explore/` | 一次性验证代码（探针）与结论。**不是正式实现，别拿去复用**。索引见 [`explore/index.md`](explore/index.md) |
| `docs/` | 设计文档：`ADR/`（决策记录）、`known-limitations/`、`agent_notes/`、`docs-rules/` |
| `tests/` | 按 `simple_rag/` 的结构镜像摆放 |
| `data/` | 语料（LifePrism 日记数据） |

各子包的细节写在各自的 `README.md` 里（如
[`simple_rag/repository/bm25/README.md`](simple_rag/repository/bm25/README.md)）。

## 当前进度

已经落地：

- **切分** —— 按标题切 Markdown，行级 + 行内切，小块可合并
- **索引** —— `RagIndexingPipeline` 四步：chunking → merge → embedding → store，
  三张表一个事务
- **向量检索** —— sqlite-vec，独立 `.db` 文件
- **关键词检索** —— BM25 三个可换实现：`fts5` / `rank_bm25` / `own_bm25`
- **综合检索** —— `RetrievalClient` 两路并发召回 + RRF 融合，回数据表补正文

还没进 `simple_rag/`：rerank 目前只有探针结论（[`explore/rerank/`](explore/rerank/)），
`retrieval/` 里没有它的位置。

## 技术选型与实测结论

结论都来自 `explore/` 里的探针实测，不是推断。

**向量库用 sqlite-vec** —— 可用。三个必须主动防的坑：dtype 写错会**静默损坏数据**；
partition key 用不了 `IN` / `OR` 也不能 UPDATE；PyInstaller 必须显式收集 `vec0.dll`。
详见 [`explore/sqlite-vec/FINDINGS.md`](explore/sqlite-vec/FINDINGS.md)。

**BM25 保留三类实现** —— 各自缺口不同，换实现是工厂的一个参数：

| 实现 | 可调 `k1` / `b` | 增量 | 语料常驻内存 |
|---|---|---|---|
| `fts5` | ✗（硬编码 1.2 / 0.75） | ✓ | 否 |
| `rank_bm25` | ✓（改参数要重建） | ✗ | **是**（5 万篇实测 1,927 MB） |
| `own_bm25` | ✓（**改参数不用重建**） | ✓ | 否 |

详见 [`simple_rag/repository/bm25/README.md`](simple_rag/repository/bm25/README.md)。

**rerank 用阿里云百炼 `qwen3.7-text-rerank`** —— 可用。注意它的分数**越大越相关**，
与本仓库「分数越小越相关」的方向相反，融合前要统一。详见
[`explore/rerank/README.md`](explore/rerank/README.md)。

**稀疏向量不做第三路** —— 实测它跟着 BM25 一起崩；词面弱时加进 RRF 反而有害。
详见 [`explore/实际测试/sparse探针/README.md`](explore/实际测试/sparse探针/README.md)。

## 怎么跑起来

```bash
pip install -e ".[dev]"     # 安装（含 pytest）
python -m pytest            # 跑测试
```

测试**不需要网络、也不需要 API key** —— 对外部 API 的调用一律走 `httpx.MockTransport`。
只有 `explore/` 里打真接口的脚本才需要 `.env`（键名见 `.env` 本身，该文件不入库）。

没有 CLI —— 入口是库 API，以及 `explore/` 下的一次性脚本。

## 设计规则

见 [CLAUDE.md](CLAUDE.md)。核心是三条：解耦化、接口化、模块化，
**不替调用方猜下游需求**。
