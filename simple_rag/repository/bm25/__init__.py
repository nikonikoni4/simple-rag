"""`simple_rag.repository.bm25` —— FTS5 关键词检索。

**与 `vec` 互不依赖。** 两边只共用调用方注入的连接（`simple_rag.db.Database`）
与分词器（`simple_rag.tokenization`）。

分词器**不落盘** —— 换分词器要全库重建是调用方的运维动作。两侧不一致时同一个词
不是同一个 token，**结果是静默零召回、不报错**。
"""

from .bm25_store import BM25Index, BM25SearchResult

__all__ = ["BM25Index", "BM25SearchResult"]
