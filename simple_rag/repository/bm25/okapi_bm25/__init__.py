"""`simple_rag.repository.bm25.okapi_bm25` —— rank_bm25 库（BM25Okapi）实现。

一切 rank_bm25 专属的东西都内聚在这里（打分、pickle 持久化）。对外不导出 ——
统一入口是 `simple_rag.repository.bm25.create_bm25("rank_bm25", ...)`，impl 键不变。

**子包名不能叫 `rank_bm25`** —— 会和所调用的库撞名（它是顶层模块
`site-packages/rank_bm25.py`）。只要 `bm25/` 这一层进了 `sys.path`（IDE run config、
`cd` 进来、工具插路径），本子包就会抢先成为顶层 `rank_bm25`，于是 `index.py` 里的
`from rank_bm25 import BM25Okapi` 变成导入自己 → ImportError。改叫 okapi_bm25
（按底层算法 BM25Okapi 命名，与 `fts5_bm25` 的风格一致）后彻底避开。
"""

from .index import RankBM25Index

__all__ = ["RankBM25Index"]
