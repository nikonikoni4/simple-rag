# simple_rag/repository/chunk_store —— 数据表

普通 SQLite 表（**不是**虚拟表）。一个 `.db` 文件一张表 `chunk_data`。

**它不认识任何业务概念** —— 存的是什么内容、来自哪些文件、要不要加标签，
都是调用方的决定。

**连接由调用方注入**（[simple_rag.db](../../db/README.md)）—— 本子包不创建、
也不关闭连接。

- 字段模型与检索取数路径：[表结构约定.md](../表结构约定.md)
- 设计规则：[CLAUDE.md](../../../CLAUDE.md)

## 它和向量表的分工

| | 向量表 [vec](../vec/README.md) | 数据表（本包） |
|---|---|---|
| 底层 | `vec0` 虚拟表 | 普通 SQLite 表 |
| 管什么 | 「哪些 chunk 跟查询像」 | 「这个 chunk 是什么、来自哪」 |
| 一个 chunk 占几行 | 1 行 | **N 行**（几个来源就几行） |
| 加字段 | 能，但要**重建表** + 重跑 embedding | 能，`ALTER TABLE` 即可，**不用重嵌** |

检索时先由向量表给出 `chunk_id`，再回这里取正文与来源。

## 基本字段

写入时**必须给**，DDL 由模块固定生成：

| 列 | 说明 |
|---|---|
| `chunk_id` | 所属 chunk。**不是主键** —— 一个 chunk 可能有多行 |
| `path` | 来源文件路径（非空） |
| `start_line` / `end_line` | 该来源在文件里的行区间（闭区间，`start <= end`） |
| `created_at` / `updated_at` | ISO 8601 + UTC，由模块填 |
| `content` | 正文。同一 chunk 的各行是同一份，**逐行重复存放** |

其余列由调用方通过 `Schema(fields=(...))` 声明，一律建成 `text`。

## 用法

```python
from simple_rag.db import Database
from simple_rag.repository.chunk_store import ChunkStore, Schema

with Database("rag.db") as database:
    store = ChunkStore(database.connection)
    store.create_table(Schema(fields=("tag",)))   # 扩展列可选，默认没有

    # 一条记录 = 一个 chunk；sources 有几项就往表里写几行
    store.insert([
        {
            "chunk_id": "abc",
            "content": "正文",
            "sources": [
                {"path": "a.md", "start_line": 1, "end_line": 10},
                {"path": "b.md", "start_line": 5, "end_line": 8},
            ],
            "tag": "日记",
        },
    ])

    rows = store.get("abc")        # 按 path / start_line 排好序，可能多行
    store.ids_by_path("a.md")      # ['abc'] —— 更新文件前先问「谁来自它」
    store.delete_by_path("a.md")   # 只清数据表这侧的行
    store.delete("abc")            # 删掉这个 chunk 的全部行
```

## 建表时一并建的两个索引

- `(chunk_id)` —— 按 id 取回一个 chunk 的全部行（检索时的回查）
- `(path)` —— 按文件反查 `chunk_id`（更新文档时先定位要动哪些块）

## 调用方要知道的约束

| | |
|---|---|
| **表上没有唯一约束** | 同一个 `chunk_id` 分两次插入来源会**并存**。要幂等得自己先清干净再写 |
| **`content` 逐行重复** | 多来源的 chunk，正文在每一行都存一份 —— 换来的是一条 SQL 按 `path` 定位 |
| **`delete_by_path` 只动数据表** | 向量表那侧要不要跟着删，由调用方决定（先 `ids_by_path` 拿 id，再去删向量表）|
| **时间戳由模块维护** | 不用传，传了报「不认识的键」。`insert` 时两列同值 |
| **行区间必填** | `start_line` / `end_line` 不能为空，且 `start <= end`；`sources` 也不能为空 |
| **没有 `update`** | 改内容 = 先 `delete` 再 `insert`。行级局部更新没有场景，不做 |
| **连接由调用方注入** | 与 `VecDB` 一样：不建、不关、不跨线程 |
| **事务边界** | `insert` 整批一个 `SAVEPOINT`（可被外层事务嵌套）；`delete` / `delete_by_path` 是**单条单事务** |

## 异常契约

| 类型 | 含义 |
|---|---|
| `ValueError` | 输入不合法（值不对、字段名不合法、`sources` 为空、`start > end`）|
| `TypeError` | 类型不对 |
| `sqlite3.Error` | 库层面的问题 |

## 文件

| 文件 | 职责 |
|---|---|
| `__init__.py` | 对外出口：`ChunkStore` / `Schema` / `ChunkRow` |
| `store.py` | `ChunkStore` —— 注入的连接、Schema、事务控制 |
| `_schema.py` | Schema 校验 + DDL + 索引生成（纯函数） |
| `_store.py` | SQL 文本构造 + 记录 / 来源校验（纯函数） |

测试在 `tests/repository/chunk_store/`（`python -m pytest tests/repository/chunk_store`）。
