# simple_rag/repository —— 存储层

三个**互不依赖**的子包：

| 子包 | 底层 | 管什么 | 说明 |
|---|---|---|---|
| [vec](vec/README.md) | sqlite-vec 的 `vec0` 虚拟表 | 向量相似度（KNN） | 只回答「哪些 chunk 跟查询像」 |
| [chunk_store](chunk_store/README.md) | 普通 SQLite 表 | 按 `chunk_id` 取正文与来源 | 回答「这个 chunk 是什么、来自哪」 |
| [bm25](bm25/README.md) | SQLite FTS5 虚拟表 | 关键词（BM25 打分） | 存 token 串的倒排索引 |

`vec` 与 `chunk_store` **成对使用**：先由向量表给出 `chunk_id`，再回数据表取内容。
两张表的字段模型（模块定基本字段，其余列由调用方声明）见
[表结构约定.md](表结构约定.md)。

**本包不导出任何名字**（`__all__ = []`）。请从子包导入：

```python
from simple_rag.repository.vec import Schema, VecDB
from simple_rag.repository.chunk_store import ChunkStore
from simple_rag.repository.bm25 import BM25Index
```

不放转发 —— 同一个类有两条公开路径时，「该用哪个」和「改哪个」都会变得含糊。

## 依赖方向

```
retrieval  →  repository.{vec, chunk_store, bm25}  →  db
                          ↑
                   tokenization（只被 bm25 用）
```

- **三个子包互不 import。** 各有一份 `_schema.py` / `_store.py`，表名常量
  （`chunks` / `chunk_data` / `chunks_fts`）也是各写各的 —— 合并成一处
  「表名常量表」等于让它们互相依赖，换来的只是省两行。
- 三个都**不创建也不关闭连接**，只接收 `simple_rag.db.Database` 注入的连接。
- 三个都**不认识业务概念** —— 字段叫什么、存什么、要不要过滤，都是调用方的决定。

## 两个子包的性质差异（不是设计缺陷）

| | `vec`（vec0） | `bm25`（FTS5） |
|---|---|---|
| 原地改一列 | ✅ 支持（探针已验证） | ❌ 只能删+插 |
| 所以有 `update` 吗 | 有 | **没有** —— 藏起「删了再插」会让人误以为它是原子的 |
| `chunk_id` 的容忍度 | 任意字符串 | 必须是 ≥15 位 hex（要转成 `rowid`） |
| 定位一行 | `chunk_id` 主键（有索引） | 按 `rowid`；按 `chunk_id` 是**全表扫** |
| 写开销 | 快 | 双写让写入慢约 68%（5 万条实测） |
| 检索耗时 | 5 万条 63.4 ms（KNN） | 5 万条 63.4 ms（`order by bm25()`，**线性**） |

## 几张表怎么保持一致

跨表写入可以放进**同一个事务**（共用同一个连接时，SQLite 的跨表事务原生原子）。
一次入库通常要同时写三张表：向量表、数据表、FTS5。

```python
with Database("rag.db") as database:
    conn = database.connection
    conn.execute("savepoint import")
    try:
        vec.insert(records)          # 向量
        chunk_store.insert(records)  # 正文 + 来源
        bm25.insert(records)         # 关键词
        conn.execute("release import")
    except BaseException:
        conn.execute("rollback to import")
        conn.execute("release import")
        raise
```

⚠️ `rollback to` 之后必须再 `release`，否则事务一直挂着不提交**且不报错**。
详见 [db/README.md](../db/README.md)。

## 文件

| 文件 | 职责 |
|---|---|
| `__init__.py` | `__all__ = []`，不导出任何名字 |
| `vec/` | 向量存储。见 [vec/README.md](vec/README.md) |
| `chunk_store/` | 数据表：正文 + 来源。见 [chunk_store/README.md](chunk_store/README.md) |
| `bm25/` | 关键词存储。见 [bm25/README.md](bm25/README.md) |
| `表结构约定.md` | 两张表的字段模型（基本字段 + 外部声明的列）与检索取数路径 |

## 测试

```bash
python -m pytest tests/repository -q
```

测试按模块结构镜像在 `tests/repository/` 下：`vec/` 对应 `vec/`，
`bm25/` 对应 `bm25/`（再往下分 `fts5_bm25/`、`okapi_bm25/`）。

⚠️ 这些测试目录**都没有 `__init__.py`**，pytest 按 basename 导入 ——
不同目录下的测试文件**不能重名**（当前无重名）。加新测试时留意。
