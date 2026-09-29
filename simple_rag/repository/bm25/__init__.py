"""`simple_rag.repository.bm25` —— BM25 关键词检索（多实现，统一接口）。

**与 `vec` 互不依赖。** 两边只共用调用方注入的连接（`simple_rag.db.Database`，
仅 fts5 实现用）与分词器（`simple_rag.tokenization`）。

对外只有三样东西：`BM25Index`（统一接口）、`BM25SearchResult`（输出形状）、
`create_bm25`（工厂）。**具体实现类不出口** —— 换实现是 `create_bm25` 的
一个参数，不是调用方 import 路径的改动。

分词器**不落盘** —— 换分词器要全库重建是调用方的运维动作。两侧不一致时同一个词
不是同一个 token，**结果是静默零召回、不报错**。

两个实现的能力差异（详见包 README）：

| | fts5 | rank_bm25 |
|---|---|---|
| 持久化 | SQLite（注入的连接） | pickle 文件（构造时给 `persist_path`）或纯内存 |
| 增量 insert / delete / replace | 支持 | **抛 `NotImplementedError`**，只能 `rebuild` |
| `k1` / `b` | 不可调（FTS5 硬编码 1.2 / 0.75） | 可调（默认 1.5 / 0.75） |
| chunk_id 形状 | >= 15 位 hex（rowid 派生） | 任意非空 str |
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from simple_rag.tokenization import Tokenizer

from .base import BM25Index, BM25SearchResult
from .fts5_bm25 import FTS5BM25Index
from .rank_bm25 import RankBM25Index

__all__ = ["BM25Index", "BM25SearchResult", "create_bm25"]


def create_bm25(
    impl: Literal["fts5", "rank_bm25"],
    *,
    tokenizer: Tokenizer,
    conn: sqlite3.Connection | None = None,
    k1: float | None = None,
    b: float | None = None,
    persist_path: str | Path | None = None,
) -> BM25Index:
    """按实现名造一个 `BM25Index`。

    Args:
        impl: `"fts5"`（SQLite FTS5，可增量、持久化在 db 文件里）或
            `"rank_bm25"`（BM25Okapi，k1/b 可调、无增量）。
        tokenizer: 分词器，写入与查询共用，两实现都要。
        conn: 仅 fts5 需要 —— 调用方注入的 SQLite 连接，实现不持有其生命周期。
        k1: 仅 rank_bm25 —— 词频饱和参数，缺省取库默认 1.5。**None 是哨兵**：
            fts5 只要显式传了（哪怕传 1.5）就报错 —— 它硬编码在 1.2，
            显式传默认值与「没传」不可区分时，静默放行就是让调用方以为
            参数生效了。
        b: 仅 rank_bm25 —— 文档长度归一化强度，缺省取库默认 0.75；同 k1。
        persist_path: 仅 rank_bm25 —— pickle 持久化路径；fts5 传了报错
            （它的持久化就是注入的那个 db 文件）。

    Returns:
        统一接口 `BM25Index` 的实例。用前先 `open()`。

    Raises:
        ValueError: impl 未知，或给了当前实现不支持的参数。
            **静默忽略被禁止** —— 配置没生效却不报，比报错糟糕得多。
    """
    builders: dict[str, Callable[[], BM25Index]] = {
        "fts5": lambda: _build_fts5(tokenizer, conn, k1, b, persist_path),
        "rank_bm25": lambda: _build_rank_bm25(tokenizer, conn, k1, b, persist_path),
    }
    builder = builders.get(impl)
    if builder is None:
        raise ValueError(f"未知的 bm25 实现：{impl!r}，可选 {sorted(builders)}")
    return builder()


def _build_fts5(
    tokenizer: Tokenizer,
    conn: sqlite3.Connection | None,
    k1: float | None,
    b: float | None,
    persist_path: str | Path | None,
) -> BM25Index:
    if conn is None:
        raise ValueError("fts5 实现需要 conn（调用方注入的 SQLite 连接）")
    unsupported = []
    if k1 is not None:
        unsupported.append(f"k1={k1!r}（FTS5 硬编码在 1.2，调不了）")
    if b is not None:
        unsupported.append(f"b={b!r}（FTS5 硬编码在 0.75，调不了）")
    if persist_path is not None:
        unsupported.append(
            f"persist_path={persist_path!r}（fts5 的持久化就是注入的 db 文件）"
        )
    if unsupported:
        raise ValueError(f"fts5 实现不支持这些参数：{'；'.join(unsupported)}")
    return FTS5BM25Index(conn, tokenizer)


def _build_rank_bm25(
    tokenizer: Tokenizer,
    conn: sqlite3.Connection | None,
    k1: float | None,
    b: float | None,
    persist_path: str | Path | None,
) -> BM25Index:
    if conn is not None:
        raise ValueError(
            f"rank_bm25 实现不需要 conn（收到 {conn!r}）—— 它是纯内存 + 可选 pickle"
        )
    return RankBM25Index(
        tokenizer,
        k1=1.5 if k1 is None else k1,
        b=0.75 if b is None else b,
        persist_path=persist_path,
    )
