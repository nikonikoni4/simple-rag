# simple_rag/db —— 数据库连接的生命周期

**只管一件事**：一个 `.db` 文件的连接由谁创建、谁持有、谁关闭。

**不认识任何表、Schema、业务概念。** 表由 [repository/vec](../repository/vec/README.md)
和 [repository/bm25](../repository/bm25/README.md) 各自建。

## 用法

```python
from simple_rag.db import Database
from simple_rag.repository.vec import VecDB, Schema
from simple_rag.repository.bm25 import create_bm25
from simple_rag.tokenization import TokenizerFactory

with Database("vec.db") as database:
    conn = database.connection           # 同一个连接，注入给两边
    vec = VecDB(conn)
    bm25 = create_bm25("fts5", tokenizer=TokenizerFactory.create("jieba"), conn=conn)
```

## 谁负责关闭

**谁创建谁负责。** `VecDB` / bm25 实现（仅 fts5 用连接）只接收注入的连接，
**没有 `close()`** —— 留一个什么都不做的 `close()` 更危险：调用方以为收尾了，
实际连接还开着。

`Database.close()` 重复调用是 no-op（`sqlite3.Connection.close()` 本身幂等，实测探针 21）。

## 事务

`Database` **不提供** `transaction()` 之类的上下文管理器。要用事务，自己发
`savepoint` / `release`：

```python
with Database("vec.db") as database:
    conn = database.connection
    conn.execute("savepoint import")
    try:
        vec.insert(records)
        bm25.insert(records)
        conn.execute("release import")
    except BaseException:
        conn.execute("rollback to import")
        conn.execute("release import")
        raise
```

两个存储**共用同一个连接**时，上面这段是原子的 —— 实测 `rollback` 对两张表同时生效
（见 [explore/bm25/TECHNICAL.md](../../explore/bm25/TECHNICAL.md) §6.2）。

⚠️ **`rollback to` 之后必须再 `release`。** `rollback to` 只把数据退回到 savepoint
的状态，savepoint 本身还留在栈上；不 `release` 的话事务一直挂着不提交，**而且不报错**。

## 连接参数（实测锁定，不要顺手改）

| 参数 | 值 | 为什么 |
|---|---|---|
| `isolation_level` | `None` | 默认的 `''` 会在 DML 前隐式 `BEGIN`，模块再发 `savepoint` 会撞 `cannot start a transaction within a transaction` |
| `enable_load_extension` | load 后**立刻关** | 实测关掉之后再 `load()` 报 `not authorized` |
| `journal_mode` | `WAL` | 持久设置；写不挡读 |
| 建连接失败 | 先 `close()` 半成品再抛 | 用局部变量，否则 `connect` 自己失败时关连接会抛 `AttributeError` 盖掉真实错误 |

## 文件

| 文件 | 职责 |
|---|---|
| `__init__.py` | 对外出口：只导出 `Database` |
| `database.py` | `Database` 类 + `open_connection` 工厂 |

测试在 `tests/db/`，6 个用例（`python -m pytest tests/db`）。

`open_connection` **不从 `__init__.py` 导出** —— 要用请走深路径
`from simple_rag.db.database import open_connection`。

> `open_connection` 的函数体原样搬自 `simple_rag/repository/_connection.py`，
> 承载 4 条实测结论（探针 21 / 01），迁移时一个字符没改。
