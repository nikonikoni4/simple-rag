# simple_rag/repository —— 向量存储模块

基于 sqlite-vec 的 `vec0` 虚拟表。一个 `.db` 文件一张表。

**它不认识任何业务概念** —— 不知道数据是什么、有多少、字段该怎么设计。
叫什么字段、要不要过滤，都是调用方的决定。

- 设计：[spec](../../docs/superpowers/specs/2026-09-25-vec-db-repository-design.md)
- 踩过的坑（全部实测）：[explore/sqlite-vec](../../explore/sqlite-vec/FINDINGS.md)
- 设计规则：[CLAUDE.md](../../CLAUDE.md)

## 用法

```python
from simple_rag.repository import VecDB, Schema

with VecDB("vec.db") as db:
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

    hits = db.search(query_vector, k=5, where={"loc": "a.md"})  # 向量相似度查询
    row = db.get("c-2")                                        # 普通查询，按 id 取一行

    db.update("c-1", vector=new_vector, fields={"content": "改过的正文"})
    db.delete("c-1")
```

`create_table` 每次开库都调是**最常规的用法** —— 表已存在时会比对 `dim` / `metric`，
不一致报 `ValueError`（不是 `if not exists` 那种静默接受）。

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
| **连接由实例持有** | 不是模块级全局，要几个开几个。一个实例不跨线程；两个实例写同一个库，第二个会等到超时后报 `database is locked` |
| **没有跨记录的事务边界** | `insert` 整批一个事务；`delete` / `update` 是**单条单事务** |

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
| `vec_db.py` | `VecDB` —— 持有连接、Schema、事务控制 |
| `_schema.py` | Schema 校验 + DDL 生成（纯函数） |
| `_store.py` | SQL 文本构造 + 记录 / `where` / `k` 的校验（纯函数） |
| `_codec.py` | 向量归一化 + dtype 统一（纯函数） |
| `_connection.py` | 建连接（加载扩展） |
| `tests/` | 纯单测 3 个 + 端到端 1 个。`python -m pytest simple_rag/repository/tests` |

内部约定两条，改代码时别踩：

1. **DDL 里字段名裸写，DML 里每个标识符加方括号。** vec0 拒绝 DDL 里一切带引号的
   标识符；而 DML 里**不能用双引号** —— SQLite 找不到列名时它会退化成字符串字面量，
   拼错列名不报错而是静默返回常量（实测探针 19 / 21）。
2. **`insert` 失败时 `except` 里必须 `rollback()`。** 实测语句失败**不会**自动中止事务，
   误调 `commit()` 会让半批数据真的落库（探针 18 §8.5 / 21 §11.5）。

完整限制清单（写开销、重索引、打包等）见 spec §12。
