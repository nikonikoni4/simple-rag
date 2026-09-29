"""`simple_rag.repository.bm25.rank_bm25` —— rank_bm25 库（BM25Okapi）实现。

一切 rank_bm25 专属的东西都内聚在这里（打分、pickle 持久化）。对外不导出 ——
统一入口是 `simple_rag.repository.bm25.create_bm25("rank_bm25", ...)`。

注意：本子包与库同名不冲突 —— Python 3 绝对导入下，本包内的
`from rank_bm25 import BM25Okapi` 解析到 site-packages 的库，不是本包。
"""

from .index import RankBM25Index

__all__ = ["RankBM25Index"]
