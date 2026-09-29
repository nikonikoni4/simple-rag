"""`simple_rag.repository.bm25.fts5_bm25` —— FTS5 实现。

FTS5 的一切都内聚在这里（表 DDL、SQL 构造、事务控制）。对外不导出 ——
统一入口是 `simple_rag.repository.bm25.create_bm25("fts5", ...)`。
"""

from .index import FTS5BM25Index

__all__ = ["FTS5BM25Index"]
