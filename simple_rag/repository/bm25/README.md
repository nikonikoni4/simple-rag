# simple_rag/repository/bm25 —— 关键词检索（统一接口，多实现）

BM25 关键词检索。对外只有三样东西：**`BM25Index`（统一接口）、
`BM25SearchResult`（输出形状）、`create_bm25`（工厂）**。具体实现类不出口 ——
换实现是工厂的一个参数，不是调用方 import 路径的改动。

三个实现：

- `fts5_bm25` —— SQLite FTS5 虚拟表，打分用内置 `bm25()`。可增量、持久化
  就在注入的 db 文件里；`k1` / `b` 被 FTS5 硬编码在 1.2 / 0.75 **不可调**
  （[已知限制](../../../docs/known-limitations/2026-09-29-fts5参数不可调.md)）。
- `okapi_bm25` —— rank_bm25 库的 `BM25Okapi`。`k1` / `b` **可调**（默认 1.5 / 0.75）；
  纯内存，无增量 —— `insert` / `delete` / `replace` 直接抛 `NotImplementedError`，
  改数据只能 `rebuild` 全量重建；可选 pickle 持久化（构造时给 `persist_path`）。
- `own_bm25` —— **自设计 SQLite 表**（文档长度表 + 倒排词频表 + 全库统计表）。
  同时具备前两者的长处：`k1` / `b` **可调且换参数不用重建**、可增量、
  语料**不常驻 Python 内存**。打分公式与存储结构都由本项目掌握。

**与 [vec](../vec/README.md) 互不依赖** —— 两边只共用调用方注入的连接
（[simple_rag.db](../../db/README.md)，fts5 与 own_bm25 用）与分词器
（[simple_rag.tokenization](../../tokenization/README.md)）。

**它不认识任何业务概念** —— 存什么文本、`chunk_id` 怎么生成，都是调用方的决定。

- 选型依据与性能实测：[explore/bm25/FINDINGS.md](../../../explore/bm25/FINDINGS.md)
- FTS5 内部结构（表布局、`block` 格式、公式）：[explore/bm25/TECHNICAL.md](../../../explore/bm25/TECHNICAL.md)
- 统一接口的取舍记录：[docs/ADR/2026-09-29-BM25多实现统一接口.md](../../../docs/ADR/2026-09-29-BM25多实现统一接口.md)

## 存的是什么

**不是原文，是分词后的 token。** FTS5 的内置 tokenizer 对中文无效 ——
`unicode61` 会把整句当成**一个** token，查 `检索` 零命中；`trigram` 又要求
查询串 ≥3 字符。所以中文必须自己切（**实测**
[FINDINGS.md §4](../../../explore/bm25/FINDINGS.md)）。rank_bm25 同理 ——
`BM25Okapi` 吃的是 token 列表。

所有写入口收的 `text` 都是**原文**，分词在实现内做；调用方不用管。

```sql
-- fts5 的表（rank_bm25 没有表，token 在内存里）
create virtual table chunks_fts using fts5(
    tokens,                 -- token 串（空格分隔）
    chunk_id unindexed,     -- 原始 id，只存不索引
    tokenize='unicode61'    -- 只按空格切，不做中文分词
)

-- own_bm25 的表：一行一个 (词, 篇) 对，不存原文也不存 token 串
create table own_bm25_docs    (doc_id, chunk_id unique, dl);      -- 本篇长度
create table own_bm25_postings(term, doc_id, tf, primary key(term, doc_id));
create table own_bm25_stats   (singleton = 1, n, total_dl);       -- 全库统计
```

`own_bm25` **不存原文、不存 token 串、不存 IDF**：`tf` 与 `dl` 是本篇内可定的
全部信息，`N` / `total_dl` 是全库统计量，写入时算不出来，只能随增删改原子维护。
这是它内存不随语料增长的原因。

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

# own_bm25：k1/b 可调 + 可增量 + 语料不常驻内存
with Database("vec.db") as database:
    bm25 = create_bm25("own_bm25", tokenizer=tok,
                       conn=database.connection, k1=1.5, b=0.75)
    bm25.open()                       # 首建三张表，之后校验；幂等
    bm25.insert(records)              # 也可 delete / replace / rebuild

    # 换参数**不用重建**：同一个库换一组 k1/b 再造一个实例即可
    tuned = create_bm25("own_bm25", tokenizer=tok,
                        conn=database.connection, k1=0.3, b=0.2)
    hits = tuned.search("检索", k=10)
```

## 三实现能力对照

| | fts5 | rank_bm25 | own_bm25 |
|---|---|---|---|
| `open()` | 首建表或校验 DDL | 有 `persist_path` 且文件在则加载，否则空索引 | 首建三张表或校验，并建 TEMP 工作表 |
| `insert` / `delete` / `replace` | 支持（各自一个事务内） | **抛 `NotImplementedError`** | 支持（各自一个 `SAVEPOINT` 内） |
| `rebuild` | 删表 + 重建 + 全量插入 | 重算 `BM25Okapi`，自动写盘（配了 `persist_path`） | 一个保存点内清空 + 流式重填（不删表） |
| 持久化 | 注入的 SQLite db 文件 | pickle 文件（纯数据，见 `okapi_bm25/persistence.py`）或纯内存 | 注入的 SQLite db 文件（三张自设计表） |
| `k1` / `b` | 不可调（FTS5 硬编码 1.2 / 0.75，传参报错） | 可调（默认 1.5 / 0.75），改参数要重建 | 可调（默认 1.5 / 0.75），**改参数不用重建** |
| 语料常驻内存 | 否 | **是**（50,000 篇实测 1,927 MB） | 否（Python 只留当前这一篇） |
| chunk_id 形状 | **≥15 位 hex**（要转 rowid） | 任意 str（空串也放行，只是永不命中） | 任意 str（空串也放行，只是永不命中） |
| 打分 | FTS5 `bm25()`（idf 钳到 1e-6） | `BM25Okapi`（idf 有 epsilon floor） | 正 IDF `log1p((N-df+0.5)/(df+0.5))`，恒为正、无 floor |

三个实现的 idf **基底相同**（`log((N-n+0.5)/(n+0.5))`），差异只在负/零 idf 的
处理与默认参数：fts5 钳 1e-6、okapi 用 `ε×平均idf`、own_bm25 用 `log1p`
（`df ≤ N` 时恒为正，不需要兜底）。所以：查询词都满足 `n < N/2`（正 idf）时
fts5 与 rank_bm25 分数逐位相同、排序一致；含 `n ≥ N/2` 的高频词时排序可能不同，
且**精确并列时打破规则不同**（own_bm25 用 `chunk_id` 升序）。统一的是接口与
分数方向，实测见 [explore/bm25/同参数一致性.md](../../../explore/bm25/同参数一致性.md)。

## 调用方要知道的约束

### 统一（base 契约）

| | |
|---|---|
| **`score` 越小越相关** ⚠️ | 方向是契约；fts5 恒为负，rank_bm25 对原始分取负、常规语料为负、退化语料（词出现在过半文档）**可能为正** —— 符号不是契约 |
| **不支持的参数显式报错** | fts5 收到 `k1` / `b` / `persist_path`、rank_bm25 收到 `conn`、own_bm25 收到 `persist_path` 都抛 `ValueError` —— 静默忽略被禁止 |
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

### own_bm25 专属

| | |
|---|---|
| **换参数不用重建** ✅ | `k1` / `b` 只进打分公式、不落盘。指向同一个库换一组参数**重新造实例**即可（没有 setter，也不改 `search` 签名）。换**分词器**才必须全库重建 |
| **必须先 `open()`** | `open()` 负责建三张表与连接局部 TEMP 工作表。跳过它直接 `search` 会报 `no such table: own_bm25_stats`（首个读的是统计表） |
| **`k1` / `b` 必须是有限值** ⚠️ | `nan` / `inf` 会让打分静默退化（`nan` 排序无意义但不报错），构造时抛 `ValueError`。`k1 = 0` 合法，此时单词贡献退化为 IDF |
| **本实现分数恒为负** | 正 IDF，没有 floor 也没有下限钳制；高频词（出现在过半文档）照样参与打分 |
| **只写这三套表** | 同库里别的表（vec0 / FTS5）不受影响，可共用同一个连接、同一个事务 |
| **表被绕过接口改过就报错** | `open` 会比对三张表与索引的 DDL；统计行被删后 `search` 抛 `ValueError`，**不退化成静默空结果** |
| **`delete` 要 str** | 非 `str` 的 `chunk_id` 抛 `TypeError` —— 否则 SQLite 的类型亲和会把 `123` 悄悄当 `'123'` 比 |

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
| `fts5_bm25/` | FTS5 实现：`index.py`（事务控制）、`_schema.py`（DDL）、`_store.py`（SQL 构造 + rowid） |
| `okapi_bm25/` | rank_bm25 实现：`index.py`、`persistence.py`（pickle 纯数据）。**目录名不能叫 `rank_bm25`** —— 与所调用的库（顶层模块）撞名，`bm25/` 一旦进 `sys.path` 就会自我导入报 ImportError |
| `own_bm25/` | 自设计 SQLite 表实现：`index.py`（分词 + 事务编排）、`_schema.py`（三张主库表 + 两张 TEMP 表的 DDL 与 open 校验）、`_store.py`（参数化读写 + 打分 SQL）。**文件名带 `own_bm25_` 前缀的测试**是为了避开 `test_schema.py` 之类的重名 —— 测试目录不是包，重名会被 pytest 当成同一个模块 |

测试在 `tests/repository/bm25/`：

- 本层（83 个用例）：跨实现的工厂测试（含 own_bm25 的参数域边界）+ 三实现的统一契约测试（参数化）+ 通用校验测试
- `fts5_bm25/`（71 个）、`okapi_bm25/`（35 个）：各实现自己的测试
- `own_bm25/`（184 个）：打分 oracle 对拍（含 `k1=0` / `1e-300` / `1e308` 边界）、
  表结构与 `open` 语义、写入与事务边界

整棵子树共 373 个用例（`python -m pytest tests/repository/bm25`）。

## 为什么 SQL 层不和 `vec` 复用

两边的 SQL 形状差太远 —— vec0 是 `[embedding] match ? ... and k = ?`，
FTS5 是 `match ? order by bm25() limit ?`；参数与返回结构也不一样。
硬抽一层只会得到到处 `if` 的壳，而每个分支还要单独测。
**只有两种查询、形状差异大，分别实现的可测性收益更大。**

## 第三类实现（`own_bm25`）为什么存在

见 [ADR 2026-10-02](../../../docs/ADR/2026-10-02-新增第三类BM25实现.md)。两个既有实现
各有一处缺口，而这两处同时需要时现有两类都顶不上：

1. **fts5 不能修改参数** —— `k1` / `b` 被 FTS5 硬编码在 1.2 / 0.75（传参直接报错）。
2. **rank_bm25 每次要把全部内容读进内存** —— `BM25Okapi` 把全库构造成常驻结构，
   50,000 篇实测 **1,927 MB**（≈38 KB/篇，随 N 严格线性）；同样数据放 SQLite 只有
   **~2 MB** 页缓存（[explore/bm25/README.md](../../../explore/bm25/README.md)）。

`own_bm25` 的表结构由本项目自己设计（不套 FTS5 的虚拟表形态），打分公式也自己落 ——
所以参数可调、统计量可增量维护、语料不必进 Python 堆。

**10 万 chunks 实测**（暖缓存，p95）：k=10、≤3 词时低命中 **2.2 ms**、中命中 **22 ms**、
高命中 **89 ms**、全命中（10 万篇全部打分）**445 ms**；`open()` 的 Python 增量恒为
**+0.16 MB**（1 万与 10 万语料相同），查询期的采样峰值约 **2~3 MB** 且随 N 持平，
对照 `rank_bm25` 同规模的 1,927 MB。
代价在别处：命中 10 万篇 + 10 个查询词（k=4096）时 p95 **1.6 s** —— 本实现
**不截断候选**，所有命中都参与打分。完整矩阵、方法与边界见
[explore/bm25/OWN_BM25_FINDINGS.md](../../../explore/bm25/OWN_BM25_FINDINGS.md)。

### 参数域与调参

| | |
|---|---|
| 默认值 | `k1 = 1.5`、`b = 0.75`（与 rank_bm25 一致；**不是** FTS5 的 1.2 / 0.75） |
| `k1` | 有限数值且 `>= 0`。`k1 = 0` 合法 —— 单词贡献退化为 IDF，`tf` 不再影响该词得分 |
| `b` | 有限数值且 `0 <= b <= 1` |
| 拒绝 | `nan` / `inf` / `bool` / 非数值 / 越界，一律 `ValueError`（`None` 是「没传」的哨兵，落默认值） |
| 调参方式 | **重新构造实例**，连接指向同一个库。没有 setter，也不改 `search` 签名 |

**换 `k1` / `b` 不需要重建索引** —— 两个参数只进打分公式，不落盘。换**分词器**
仍然必须全库重建：token 是分词器的输出，两侧不一致就是静默零召回。

## 未验证

- 分词质量、与向量检索的**融合策略**（本模块只负责存与查）
- jieba 的未登录词、自定义词典
- 停用词表能否降低查询延迟（命中集变小 → `order by bm25()` 更快）

（原文里「rank_bm25 在大语料下的内存占用与查询延迟」一条已于 2026-10-02 实测闭合，
见上节与 [explore/bm25/README.md](../../../explore/bm25/README.md)。）
