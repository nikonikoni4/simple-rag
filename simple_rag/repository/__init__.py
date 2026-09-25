"""`simple_rag.repository` —— 独立的向量存储模块。

**不与任何业务概念耦合**：不知道数据是什么、有多少、字段该怎么设计。

对外只有三个名字：

- `VecStore` —— 开库、建表、增删改查
- `Schema` —— 表的形状
- `VecSearchResult` —— `search` / `get` 的返回
"""

from ._schema import Schema
from .vec_db import VecSearchResult, VecStore

__all__ = ["Schema", "VecSearchResult", "VecStore"]
