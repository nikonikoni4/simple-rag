"""`simple_rag.repository.vec` —— 向量存储。

基于 sqlite-vec 的 `vec0` 虚拟表。**不与任何业务概念耦合**：不知道数据是什么、
有多少、字段该怎么设计。

对外五个名字：

- `VecDB` —— 建表、增删改查
- `Schema` —— 表的形状
- `VecSearchResult` —— `search` / `get` 的返回
- `TABLE_NAME` —— 表名
- `vector_to_sqlite_vector` —— 向量编码（归一化 + dtype + NaN 检查）

后两个是给**跨表写 SQL 的编排层**用的（`simple_rag.indexing` 要拼
`chunk_id in (select chunk_id from 数据表 where path = ?)`）。单独用本包不需要它们。

**连接由调用方注入**（`simple_rag.db.Database`）—— 本包不创建、也不关闭它。
"""

from ._codec import vector_to_sqlite_vector
from ._schema import TABLE_NAME, Schema
from .vec_db import VecSearchResult, VecDB

__all__ = [
    "TABLE_NAME",
    "Schema",
    "VecSearchResult",
    "VecDB",
    "vector_to_sqlite_vector",
]
