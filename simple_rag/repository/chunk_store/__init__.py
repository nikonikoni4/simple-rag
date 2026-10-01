"""`simple_rag.repository.chunk_store` —— 数据表存储。

存 chunk 的正文、来源（`path` + 行区间）与调用方声明的扩展列。
**不与任何业务概念耦合**：不知道数据是什么、来自哪些文件、要存什么标签。

它与 `repository/vec` **并列**，不互相依赖：

- `vec`（向量表）管「哪些 chunk 跟查询像」—— 只做 KNN
- `chunk_store`（数据表）管「一个 chunk 是什么内容、来自哪里」

检索时先用向量表拿到 `chunk_id`，再回到这里取正文与来源。
两者同属一个连接，写入可以用同一个事务圈起来。
见 `simple_rag/repository/表结构约定.md`。

对外四个名字：

- `ChunkStore` —— 建表、增删改查
- `Schema` —— 调用方要加的扩展列
- `ChunkRow` —— `get` 的返回
- `TABLE_NAME` —— 表名。给**跨表写 SQL 的编排层**用（`simple_rag.embedding`
  要拼 `chunk_id in (select chunk_id from 本表 where path = ?)`）

**连接由调用方注入**（`simple_rag.db.Database`）—— 本包不创建、也不关闭它。
"""

from ._schema import TABLE_NAME, Schema
from .store import ChunkRow, ChunkStore

__all__ = ["TABLE_NAME", "ChunkRow", "ChunkStore", "Schema"]
