# simple_rag/repository/bm25 —— 关键词检索（统一接口，多实现）

BM25 关键词检索。对外只有三样东西：**`BM25Index`（统一接口）、
`BM25SearchResult`（输出形状）、`create_bm25`（工厂）**。具体实现类不出口 ——
换实现是工厂的一个参数，不是调用方 import 路径的改动。

两个实现：

- `fts5_bm25` —— SQLite FTS5 虚拟表，打分用内置 `bm25()`。可增量、持久化
  就在注入的 db 文件里；`k1` / `b` 被 FTS5 硬编码在 1.2 / 0.75 **不可调**
  （[已知限制](../../../../docs/known-limitations/2026-09-29-fts5参数不可调.md)）。
- `rank_bm25` —— rank_bm25 库的 `BM25Okapi`。`k1` / `b` **可调**（默认 1.5 / 0.75）；
  纯内存，无增量 —— `insert` / `delete` / `replace` 直接抛 `NotImplementedError`，
  改数据只能 `rebuild` 全量重建；可选 pickle 持久化（构造时给 `persist_path`）。

**与 [vec](../vec/README.md) 互不依赖** —— 两边只共用调用方注入的连接
（[simple_rag.db](../../db/README.md)，仅 fts5 用）与分词器
（[simple_rag.tokenization](../../tokenization/README.md)）。

**它不认识任何业务概念** —— 存什么文本、`chunk_id` 怎么生成，都是调用方的决定。

- 选型依据与性能实测：[explore/bm25/FINDINGS.md](../../../../explore/bm25/FINDINGS.md)
- FTS5 内部结构（表布局、`block` 格式、公式）：[explore/bm25/TECHNICAL.md](../../../../explore/bm25/TECHNICAL.md)
- 统一接口的取舍记录：[docs/ADR/2026-09-29-BM25多实现统一接口.md](../../../../docs/ADR/2026-09-29-BM25多实现统一接口.md)

## 存的是什么

**不是原文，是分词后的 token。** FTS5 的内置 tokenizer 对中文无效 ——
`unicode61` 会把整句当成**一个** token，查 `检索` 零命中；`trigram` 又要求
查询串 ≥3 字符。所以中文必须自己切（**实测**
[FINDINGS.md §4](../../../../explore/bm25/FINDINGS.md)）。rank_bm25 同理 ——
`BM25Okapi` 吃的是 token 列表。

所有写入口收的 `text` 都是**原文**，分词在实现内做；调用方不用管。

```sql
-- fts5 的表（rank_bm25 没有表，token 在内存里）
create virtual table chunks_fts using fts5(
    tokens,                 -- token 串（空格分隔）
    chunk_id unindexed,     -- 原始 id，只存不索引
    tokenize='unicode61'    -- 只按空格切，不做中文分词
)
```

## 用法

```python
from simple_rag.db import Database
from simple_rag.repository.bm25 import create_bm25
from simple_rag.tokenization import TokenizerFactory

tok = TokenizerFactory.create("jieba")

# fts5：可增量，持久化 = 注入的 db 文件
with Database("vec.db") as database:
    bm25 = create_bm25("fts5", tokenizer=tok, conn=database.connection)
    bm25.open()                       # 首建或校验表，幂等

    bm25.insert([
        {"chunk_id": "0123...ef", "text": "向量检索是基于语义的检索方法"},
    ])
    bm25.replace([{"chunk_id": "0123...ef", "text": "改过的正文"}])  # 覆盖或插入
    bm25.rebuild(records)             # 全量重建（删表重来）
    bm25.delete(["0123...ef"])

    hits = bm25.search("检索", k=10)
    # [BM25SearchResult(chunk_id='0123...ef', score=-7.4...)]

# rank_bm25：k1/b 可调，纯内存 + 可选 pickle
bm25 = create_bm25("rank_bm25", tokenizer=tok, k1=1.2, b=0.75,
                   persist_path="bm25.pkl")
bm25.open()                           # 文件在则加载，不在则空索引
bm25.rebuild(records)                 # 唯一的写入口；自动写盘
hits = bm25.search("检索", k=10)      # 分数同样是「越小越相关」
```

## 两实现能力对照

| | fts5 | rank_bm25 |
|---|---|---|
| `open()` | 首建表或校验 DDL | 有 `persist_path` 且文件在则加载，否则空索引 |
| `insert` / `delete` / `replace` | 支持（各自一个事务内） | **抛 `NotImplementedError`** |
| `rebuild` | 删表 + 重建 + 全量插入 | 重算 `BM25Okapi`，自动写盘（配了 `persist_path`） |
| 持久化 | 注入的 SQLite db 文件 | pickle 文件（纯数据，见 `okapi_bm25/persistence.py`）或纯内存 |
| `k1` / `b` | 不可调（FTS5 硬编码 1.2 / 0.75，传参报错） | 可调（默认 1.5 / 0.75） |
| chunk_id 形状 | **≥15 位 hex**（要转 rowid） | 任意 str（空串也放行，只是永不命中） |
| 打分 | FTS5 `bm25()`（idf 钳到 1e-6） | `BM25Okapi`（idf 有 epsilon floor） |

两边的 idf 公式与默认参数不同 —— **分数和排序不逐位可比**，统一的是接口与
方向（见下）。

## 调用方要知道的约束

### 统一（base 契约）

| | |
|---|---|
| **`score` 越小越相关** ⚠️ | 方向是契约；fts5 恒为负，rank_bm25 对原始分取负、常规语料为负、退化语料（词出现在过半文档）**可能为正** —— 符号不是契约 |
| **不支持的参数显式报错** | fts5 收到 `k1` / `b` / `persist_path`、rank_bm25 收到 `conn` 都抛 `ValueError` —— 静默忽略被禁止 |
| **只返回命中文档** | 查询词一个都不在文中的文档不进结果，即使 k 给得很大 |
| **换分词器要全库重建** | 写入侧和查询侧不一致时，同一个词不是同一个 token —— **静默零召回、不报错**。升级 jieba 也算换分词器 |
| **批内重复提前拦** | `prepare_rows` 在写库前抛 `ValueError` 并指出是哪两条 |

### fts5 专属

| | |
|---|---|
| **`order by bm25()` 是线性的** ⚠️ | 要给**全部命中文档**打分，耗时随命中集线性增长，**不是 O(log N)**。5 万条 63.4 ms（全量现算的方案是 2.115 s，已否决） |
| **rowid 是 60 bit 截断** ⚠️ | 两个 `chunk_id` 若**前 15 位 hex 相同**，第二条会撞主键、抛 `ValueError`。5 万条实测无碰撞，但这是概率不是保证 |
| **没有 `close()`** | 连接归调用方（`Database`）。留一个什么都不做的 `close()` 会让人以为收尾了 |
| **写入慢约 68%** | 双写（vec0 + FTS5）的代价，一次性。5 万条 1.873 s → 3.148 s |

### rank_bm25 专属

| | |
|---|---|
| **没有增量** | `BM25Okapi` 构造即终态。不做「写盘后全量重算」式的伪增量 —— 一次插入一次 O(N) 重算是性能陷阱，宁可明确报错 |
| **pickle 只信自己写的文件** | 反序列化可执行任意代码。文件里是纯数据（chunk_id + token），带格式版本号，坏文件 / 版本不认抛 `ValueError` |
| **不给 `persist_path` 就不落盘** | 重启后调用方自己拿全量 records `rebuild` |

## 和 `vec` 配合（fts5）

两边**共用同一个连接**时，写入可以放进同一个事务（SQLite 的跨表事务原生原子）：

```python
conn.execute("savepoint import")
try:
    vec.insert(records)     # 内部也是 savepoint，会嵌套成子事务
    bm25.insert(records)
    conn.execute("release import")
except BaseException:
    conn.execute("rollback to import")
    conn.execute("release import")
    raise
```

⚠️ `rollback to` 之后必须再 `release`，否则事务一直挂着不提交**且不报错**。

## 文件

| 文件 | 职责 |
|---|---|
| `__init__.py` | 对外出口：`BM25Index` / `BM25SearchResult` / `create_bm25` 工厂 |
| `base/` | 实现无关层：`BM25Index` ABC、`BM25SearchResult`、通用校验（纯函数） |
| `fts5_bm25/` | FTS5 实现：`index.py`（事务控制）、`_schema.py`（DDL）、`_store.py`（SQL 构造 + rowid）、`tests/` |
| `okapi_bm25/` | rank_bm25 实现：`index.py`、`persistence.py`（pickle 纯数据）、`tests/`。**目录名不能叫 `rank_bm25`** —— 与所调用的库（顶层模块）撞名，`bm25/` 一旦进 `sys.path` 就会自我导入报 ImportError |
| `tests/` | 跨实现：工厂测试 + 两实现的统一契约测试（参数化）+ 通用校验测试。共 148 个用例 |

`python -m pytest simple_rag/repository/bm25`

## 为什么 SQL 层不和 `vec` 复用

两边的 SQL 形状差太远 —— vec0 是 `[embedding] match ? ... and k = ?`，
FTS5 是 `match ? order by bm25() limit ?`；参数与返回结构也不一样。
硬抽一层只会得到到处 `if` 的壳，而每个分支还要单独测。
**只有两种查询、形状差异大，分别实现的可测性收益更大。**

## 未验证

- 分词质量、与向量检索的**融合策略**（本模块只负责存与查）
- jieba 的未登录词、自定义词典
- 停用词表能否降低查询延迟（命中集变小 → `order by bm25()` 更快）
- rank_bm25 实现在大语料下的内存占用与查询延迟（`get_scores` 是全量打分）
