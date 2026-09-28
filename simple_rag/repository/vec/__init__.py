"""`simple_rag.repository.vec` —— 向量存储。

基于 sqlite-vec 的 `vec0` 虚拟表。**不与任何业务概念耦合**：不知道数据是什么、
有多少、字段该怎么设计。

对外只有三个名字：

- `VecDB` —— 建表、增删改查
- `Schema` —— 表的形状
- `VecSearchResult` —— `search` / `get` 的返回

**连接由调用方注入**（`simple_rag.db.Database`）—— 本包不创建、也不关闭它。
"""

from ._schema import Schema
from .vec_db import VecSearchResult, VecDB

__all__ = ["Schema", "VecSearchResult", "VecDB"]
