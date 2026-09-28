# simple_rag/repository/bm25 —— FTS5 关键词检索

基于 SQLite **FTS5** 虚拟表的关键词检索，打分用 FTS5 内置的 `bm25()`。

**与 [vec](../vec/README.md) 互不依赖** —— 两边只共用调用方注入的连接
（[simple_rag.db](../../db/README.md)）与分词器
（[simple_rag.tokenization](../../tokenization/README.md)）。

**它不认识任何业务概念** —— 存什么文本、`chunk_id` 怎么生成，都是调用方的决定。

- 选型依据与性能实测：[explore/bm25/FINDINGS.md](../../../../explore/bm25/FINDINGS.md)
- FTS5 内部结构（表布局、`block` 格式、公式）：[explore/bm25/TECHNICAL.md](../../../../explore/bm25/TECHNICAL.md)

## 存的是什么

**不是原文，是分词后的 token 串。**

FTS5 的内置 tokenizer 对中文无效 —— `unicode61` 会把整句当成**一个** token，
查 `检索` 零命中；`trigram` 又要求查询串 ≥3 字符。所以中文必须自己切
（**实测** [FINDINGS.md §4](../../../../explore/bm25/FINDINGS.md)）。

`insert` 收的 `text` 是**原文**，分词在类内做；调用方不用管。

```sql
create virtual table chunks_fts using fts5(
    tokens,                 -- token 串（空格分隔）
    chunk_id unindexed,     -- 原始 id，只存不索引
    tokenize='unicode61'    -- 只按空格切，不做中文分词
)
```

## 用法

```python
from simple_rag.db import Database
from simple_rag.repository.bm25 import BM25Index
from simple_rag.tokenization import TokenizerFactory

with Database("vec.db") as database:
    bm25 = BM25Index(database.connection, TokenizerFactory.create("jieba"))
    bm25.create_table()

    bm25.insert([
        {"chunk_id": "0123...ef", "text": "向量检索是基于语义的检索方法"},
    ])

    hits = bm25.search("检索", k=10)
    # [BM25SearchResult(chunk_id='0123...ef', score=-7.4...)]

    bm25.delete(["0123...ef"])
```

`chunk_id` 必须是 **≥15 位 hex**（要转成 rowid），本项目的
`Chunk.chunk_id` 是 `blake2b(digest_size=16).hexdigest()`（32 位），天然满足。

## 调用方要知道的约束

| | |
|---|---|
| **`score` 是负数** ⚠️ | `bm25()` 整体取负，**越小越相关**。结果已按它升序（最相关在前）。取负 / 归一化 / RRF 融合是检索层的决策，本模块不做 |
| **`order by bm25()` 是线性的** ⚠️ | 要给**全部命中文档**打分，耗时随命中集线性增长，**不是 O(log N)**。5 万条 63.4 ms（全量现算的方案是 2.115 s，已否决） |
| **rowid 是 60 bit 截断** ⚠️ | 两个 `chunk_id` 若**前 15 位 hex 相同**，第二条会撞主键、抛 `ValueError`。5 万条实测无碰撞，但这是概率不是保证 |
| **换分词器要全库重建** ⚠️ | 写入侧和查询侧不一致时，同一个词不是同一个 token —— **静默零召回、不报错**。升级 jieba 也算换分词器 |
| **没有 `update()`** | FTS5 表不支持原地改列，改 = 删 + 插。给一个 `update` 只是把两步藏起来，还让人误以为它是原子的 |
| **没有 `close()`** | 连接归调用方（`Database`）。留一个什么都不做的 `close()` 会让人以为收尾了 |
| **`chunk_id` 必须是 hex** | 与 `vec` 不同 —— vec0 接受任意字符串，这里要转成整数 rowid。「是 hex 但不足 15 位」会抛 `ValueError`（超短会静默缩小 rowid 空间、碰撞概率上升） |
| **批内重复提前拦** | `prepare_rows` 在写库前抛 `ValueError` 并指出是哪两条 |
| **写入慢约 68%** | 双写（vec0 + FTS5）的代价，一次性。5 万条 1.873 s → 3.148 s |

## 和 `vec` 配合

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
| `__init__.py` | 对外出口：`BM25Index` / `BM25SearchResult` |
| `bm25_store.py` | `BM25Index` —— 注入的连接与分词器、事务控制 |
| `_schema.py` | FTS5 DDL 生成 + 校验（纯函数） |
| `_store.py` | SQL 文本构造 + `rowid` / 记录 / `k` 的校验（纯函数） |
| `tests/` | 77 个用例。`python -m pytest simple_rag/repository/bm25` |

## 为什么 SQL 层不和 `vec` 复用

两边的 SQL 形状差太远 —— vec0 是 `[embedding] match ? ... and k = ?`，
FTS5 是 `match ? order by bm25() limit ?`；参数与返回结构也不一样。
硬抽一层只会得到到处 `if` 的壳，而每个分支还要单独测。
**只有两种查询、形状差异大，分别实现的可测性收益更大。**

## 未验证

- 分词质量、与向量检索的**融合策略**（本模块只负责存与查）
- jieba 的未登录词、自定义词典
- 停用词表能否降低查询延迟（命中集变小 → `order by bm25()` 更快）
