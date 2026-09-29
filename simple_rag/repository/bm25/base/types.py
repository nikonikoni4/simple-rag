"""BM25 实现无关的公共类型。

`BM25SearchResult` 是所有实现共用的输出形状 —— 统一接口的调用方只看得到它，
不感知背后是 FTS5 还是 rank_bm25。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BM25SearchResult:
    """一条命中。

    **`score` 越小越相关**（方向是契约），结果已按它升序（最相关在前）。
    fts5 实现直接输出 FTS5 `bm25()` 的原始分，恒为负；rank_bm25 实现对
    `BM25Okapi` 的原始分**取负**，常规语料下为负，退化语料（词出现在过半
    文档，idf 被 epsilon floor 成负数）下可能为正 —— **符号不是契约**。
    取绝对值 / 归一化 / RRF 融合是检索层的排序决策，存储层不替它拍板。
    """

    chunk_id: str
    score: float
