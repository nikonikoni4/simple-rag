"""`simple_rag.repository.bm25.own_bm25` —— 自设计 SQLite 表的第三类 BM25 实现。

一切内聚在这里：表 DDL 与 open 校验（`_schema.py`）、参数化读写与打分 SQL
（`_store.py`）、分词与事务编排（`index.py`）。对外不导出 —— 统一入口是
`simple_rag.repository.bm25.create_bm25("own_bm25", ...)`。
"""

from .index import OwnBM25Index

__all__ = ["OwnBM25Index"]
