# simple_rag/repository/vec —— 向量存储

基于 sqlite-vec 的 `vec0` 虚拟表。一个 `.db` 文件一张表。

**它不认识任何业务概念** —— 不知道数据是什么、有多少、字段该怎么设计。
叫什么字段、要不要过滤，都是调用方的决定。

**连接由调用方注入**（[simple_rag.db](../../db/README.md)）—— 本子包不创建、
也不关闭连接。

- 设计：[spec](../../../../docs/superpowers/specs/2026-09-25-vec-db-repository-design.md)
- 踩过的坑（全部实测）：[explore/sqlite-vec](../../../../explore/sqlite-vec/FINDINGS.md)
- 设计规则：[CLAUDE.md](../../../../CLAUDE.md)

## sqlite-vec 的列类型与本模块的对应

底层 `vec0` 虚拟表有 **5 类列**，本模块只用了其中 **4 类**（不用 partition key）：

| sqlite-vec 列类型 | 它的作用 | 本模块怎么用 |
|---|---|---|
| **向量列** | 存向量；KNN 靠它算距离 | 固定列 `embedding float[dim] distance_metric=…` |
| **主键列** | 唯一标识；库强制唯一、自带索引 | 固定列 `chunk_id text primary key`（值由调用方给） |
| **metadata 列**（普通列） | 能 SELECT，**能进 KNN 的 `where` 过滤** | `created_at` / `updated_at` + **`filterable` 里声明的字段** |
| **辅助列**（加号列） | 能 SELECT，适合放长文本；**不能进 KNN 的 `where`** | **不在 `filterable` 里的字段**（DDL 里写作 `+字段名`） |
| **partition key** | 分区加速，但查询形状受限（`IN`/`OR` 会崩、不能 UPDATE） | **不使用** |

要点：

- **`filterable` 的语义就一句话**：把字段存成 **metadata 列**（而不是辅助列）。存成 metadata 才有资格进 `search` 的 `where`。
- **两类列的 16 个上限各自独立、互不叠加**；`created_at` / `updated_at` 占掉 2 个 metadata 名额，所以 `filterable` 最多 **14** 个、其余字段最多 **16** 个。
- **`chunk_id` 主键不占** metadata 名额。
- 列类型的完整实测差异见 [FINDINGS.md §4](../../../../explore/sqlite-vec/FINDINGS.md)。

## 用法

```python
from simple_rag.db import Database
from simple_rag.repository.vec import Schema, VecDB

with Database("vec.db") as database:
    db = VecDB(database.connection)     # 连接是注入的，本类不建也不关

    # "loc" / "content" 是随手取的名字，模块不认识它们代表的任何含义
    db.create_table(Schema(
        dim=768,
        metric="cosine",        # 没有默认值 —— 建表时锁死，之后改不了
        fields=("loc", "content"),
        filterable=("loc",),    # 从 fields 里挑出哪些能进 search 的 where
    ))

    db.insert([
        {"chunk_id": "c-1", "vector": [...], "loc": "a.md", "content": "..."},
        {"chunk_id": "c-2", "vector": [...], "loc": "b.md", "content": "..."},
    ])

    # 向量相似度查询：where 的值可为 str（等值）或 (操作符, 值) 元组（范围）
    hits = db.search(query_vector, k=5, where={"loc": "a.md"})
    hits = db.search(query_vector, k=5, where={"loc": (">=", "2025-01-01")})  # 时间范围
    row = db.get("c-2")                                        # 普通查询，按 id 取一行

    db.update("c-1", vector=new_vector, fields={"content": "改过的正文"})
    db.delete("c-1")
```

`create_table` 每次开库都调是**最常规的用法** —— 表已存在时会比对 `dim` / `metric`，
不一致报 `ValueError`（不是 `if not exists` 那种静默接受）。

## `search` 的 `where`

- 键只能是 `chunk_id`，或 `filterable` 里声明的字段（其余字段存成加号列，**不能进 `where`**）
- 值两种写法：**`str` = 等值**；**`(操作符, 值)` = 比较**，操作符为 `>` / `>=` / `<` / `<=`（`=` 亦可）
- 多个键之间是 **AND**；`None` 或空 `dict` = 不过滤
- ⚠️ **字段列都是 `TEXT`，比较是字符串序**：对 ISO 时间戳（等长，字典序 = 时间序）正确；
  对纯数字**不正确**（`"9" >= "18"` 为真）。要按数字比，请补零成等长

## 两条查询路径

**这是本模块的核心，不能混。** 返回类型是同一个 `VecSearchResult`，`get` 的 `distance` 是 `None`。

| | `search` 向量相似度查询 | `get` 普通查询 |
|---|---|---|
| 要查询向量 | ✅ 必须 | ❌ 不需要 |
| 要 `k` | ✅ 必须（`0 <= k <= 4096`）| ❌ 没有这个概念 |
| 返回排序 | 按距离升序 | 不按距离 |
| 返回条数 | 最多 `k` 条 | 取到几行是几行 |
| 加号列 | 能取到、**不能在 `where` 里过滤** | 能取到、不过滤 |

两条路径**不能互相替代**：`search` 必须有查询向量、必须给 `k`，而且过滤发生在取
Top-K **之前** —— 所以它拿不到「某个字段值下的全部结果」，除非把 `k` 开到足够大。
反过来，按 id 精确取一行也只能用 `get`。

## 调用方要知道的约束

| | |
|---|---|
| **可过滤字段的值应当 ≤ 12 字符** ⚠️ | 超了**结果仍然正确，但慢约 20 倍**（实测硬边界，不是渐进劣化）。路径 / URL / UUID 这类天生超长的值不适合当 `filterable` |
| **两类列各自 16 个上限** | 普通列 16（`created_at` / `updated_at` 占 2 个）→ `filterable` 最多 14；加号列最多 16 |
| **字段名** | 字母开头，后跟字母 / 数字 / 下划线。`chunk_id` / `embedding` / `distance` / `k` / `created_at` / `updated_at` / `vector` 是保留名 |
| **`chunk_id` 是主键** | 必须由调用方给，库强制唯一，所有操作按它寻址。**重复插入报 `ValueError`**（vec0 抛的是 `OperationalError`，模块翻译过） |
| **`created_at` / `updated_at` 由模块维护** | 不用传，传了报错。`insert` 时两者同值，`update` 只刷新 `updated_at`。格式 ISO 8601 + UTC |
| **向量无条件归一化** | 统一在数据库模块内进行，不由调用方负责。dtype 也统一转 `float32` |
| **不返回向量** | `get` / `search` 都只有文本。已知代价：加字段要重建表时读不回旧向量，只能重跑 embedding |
| **连接由调用方注入** | `simple_rag.db.Database` 创建并负责关闭，`VecDB` 只接收。一个实例不跨线程；两个实例写同一个库，第二个会等到超时后报 `database is locked` |
| **事务边界** | `insert` 整批一个 `SAVEPOINT`（可被外层事务嵌套）；`delete` / `update` 是**单条单事务** |

## 异常契约

| 类型 | 含义 |
|---|---|
| `ValueError` | 输入不合法（值不对、范围不对、字段名不合法、`chunk_id` 冲突）|
| `TypeError` | 类型不对 |
| `sqlite3.Error` | 库层面的问题 |

## 文件

| 文件 | 职责 |
|---|---|
| `__init__.py` | 对外出口：`VecDB` / `Schema` / `VecSearchResult` |
| `vec_db.py` | `VecDB` —— 注入的连接、Schema、事务控制 |
| `_schema.py` | Schema 校验 + DDL 生成（纯函数） |
| `_store.py` | SQL 文本构造 + 记录 / `where` / `k` 的校验（纯函数） |
| `_codec.py` | 向量归一化 + dtype 统一（纯函数） |
| `tests/` | 纯单测 3 个 + 端到端 1 个。`python -m pytest simple_rag/repository/vec/tests` |

建连接那一层**已移到 [simple_rag/db](../../db/database.py)** —— 本子包不再自己建连接。

内部约定两条，改代码时别踩：

1. **DDL 里字段名裸写，DML 里每个标识符加方括号。** vec0 拒绝 DDL 里一切带引号的
   标识符；而 DML 里**不能用双引号** —— SQLite 找不到列名时它会退化成字符串字面量，
   拼错列名不报错而是静默返回常量（实测探针 19 / 21）。
2. **`insert` 失败时 `except` 里必须 `rollback to` + `release` 这个 savepoint。**
   实测语句失败**不会**自动中止事务，误当真提交会让半批数据真的落库
   （探针 18 §8.5 / 21 §11.5）；而只 `rollback to` 不 `release` 的话事务一直挂着
   不提交，**且不报错**。

完整限制清单（写开销、重索引、打包等）见 spec §12。
