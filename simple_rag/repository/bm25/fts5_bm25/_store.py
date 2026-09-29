"""FTS5 专属的 SQL 文本构造 + `rowid` 派生（纯函数，不碰连接）。

通用的记录 / `k` 校验在 `..base.validation`；这里只剩 FTS5 独有的东西。

**为什么 `vec` 和 `bm25` 各有一份，而不是共用一个通用层**：两边的 SQL 形状差太远
—— vec0 是 `[embedding] match ? ... and k = ?`，FTS5 是 `match ? order by bm25()
limit ?`；参数和返回结构也不一样。硬抽一层只会得到到处 `if` 的壳，而每个分支
还要单独测。**两种查询、形状差异大，分别实现的可测性收益更大。**
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ._schema import CHUNK_ID_COLUMN, FTS_TABLE_NAME, TOKENS_COLUMN

# 取 `chunk_id` 的前 15 个 hex 字符当 FTS5 的 rowid（60 bit）。
# 实测 5 万条无碰撞。用 `chunk_id` 直接定位是**全表扫**（最坏 17.7 ms），
# 用 rowid 恒定 5 µs —— 见 explore/bm25/FINDINGS.md 的维护性能表。
ROWID_HEX_CHARS = 15
ROWID_RE = re.compile(rf"^[0-9a-fA-F]{{{ROWID_HEX_CHARS},}}$")


def rowid_from_chunk_id(chunk_id: str) -> int:
    """`chunk_id` 的前 15 个 hex 字符 → 整数，作为 FTS5 的 `rowid`。

    **要求 `chunk_id` 至少 15 位 hex。** 非 hex 时 `int()` 自己会抛，但
    「是 hex 却不足 15 位」是**静默**的 —— rowid 空间变小、碰撞概率上升
    （8 位 hex 时只有 32 bit 空间，5 万条约 29% 概率撞），所以这里显式拦。

    本项目的 `Chunk.chunk_id` 是 `blake2b(digest_size=16).hexdigest()`（32 位），
    天然满足。rank_bm25 实现没有这个要求 —— 差异见包 README。

    ⚠️ 60 bit 是**截断**：两个 `chunk_id` 若前 15 位相同，后写入的会**覆盖**先写入的
    （rowid 是 FTS5 的主键）。5 万条实测无碰撞，但这是概率不是保证。
    """
    if not isinstance(chunk_id, str):
        raise TypeError(f"chunk_id 必须是 str，收到 {type(chunk_id).__name__}")
    if not ROWID_RE.match(chunk_id):
        raise ValueError(
            f"chunk_id 必须是至少 {ROWID_HEX_CHARS} 位的 hex 串，收到 {chunk_id!r}。"
            f"本项目的 Chunk.chunk_id 是 blake2b(digest_size=16).hexdigest()"
        )
    return int(chunk_id[:ROWID_HEX_CHARS], 16)


def build_ddl_drop() -> str:
    """`rebuild` 用的删表语句。

    `if exists` 在这里是对的 —— `rebuild` 的语义就是「不管表在不在，重建后只有
    这批数据」，表不在也算一种「旧状态」，照常往下走。
    """
    return f"drop table if exists {FTS_TABLE_NAME}"


def build_insert() -> str:
    """显式给 `rowid` —— FTS5 默认自增，那样就对不上 `chunk_id` 了。"""
    return (
        f"insert into {FTS_TABLE_NAME} (rowid, {TOKENS_COLUMN}, {CHUNK_ID_COLUMN}) "
        f"values (?, ?, ?)"
    )


def build_delete_by_rowids(n: int) -> str:
    """一条 SQL 批量删。

    实测（5 万条库删 100 条）：一条 `IN` 按 `rowid` 0.011 s，
    逐条按 `chunk_id` 1.400 s —— **127 倍**。
    """
    if n <= 0:
        raise ValueError(f"至少删一行，收到 n={n}（空 IN () 是 SQL 语法错误）")
    placeholders = ", ".join("?" for _ in range(n))
    return f"delete from {FTS_TABLE_NAME} where rowid in ({placeholders})"


def build_search() -> str:
    """`order by bm25(...)` 用**默认升序**。

    FTS5 的 `bm25()` 返回**负数**，越小越相关 —— 所以升序才是「最相关在前」。
    """
    return (
        f"select {CHUNK_ID_COLUMN}, bm25({FTS_TABLE_NAME}) as score "
        f"from {FTS_TABLE_NAME} "
        f"where {FTS_TABLE_NAME} match ? "
        f"order by score limit ?"
    )


def build_match_query(tokens: Sequence[str]) -> str:
    """token 序列 → FTS5 的 `match` 查询串。

    **每个 token 必须加双引号。** 不加的话，token 里的 `-` / `:` / `*` / `AND`
    会被当成 FTS5 的查询语法，轻则语法错误、重则静默换个语义。

    token 之间用 `OR`：每个词独立贡献分数，命中越全分越高 —— 这是 BM25 的用法。
    用 `AND` 会要求全部命中，召回太窄。

    token 内部的 `"` 写成 `""` 转义。
    """
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in tokens)
