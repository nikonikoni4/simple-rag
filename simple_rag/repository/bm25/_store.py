"""SQL 文本构造 + 记录 / `k` 的校验（纯函数，不碰连接）。

**为什么 `vec` 和 `bm25` 各有一份，而不是共用一个通用层**：两边的 SQL 形状差太远
—— vec0 是 `[embedding] match ? ... and k = ?`，FTS5 是 `match ? order by bm25()
limit ?`；参数和返回结构也不一样。硬抽一层只会得到到处 `if` 的壳，而每个分支
还要单独测。**两种查询、形状差异大，分别实现的可测性收益更大。**
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence

from ._schema import CHUNK_ID_COLUMN, FTS_TABLE_NAME, TOKENS_COLUMN

# 实测边界（探针 18）：4097 -> `k value in knn query too large ... the limit is 4096`
K_MIN, K_MAX = 0, 4096

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
    天然满足。

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


def check_k(k) -> int:
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k 必须是 int，收到 {type(k).__name__}")
    if not K_MIN <= k <= K_MAX:
        raise ValueError(f"k 必须在 {K_MIN}~{K_MAX} 之间，收到 {k}")
    return k


def check_record(record) -> tuple[str, str]:
    """校验一条记录的形状，返回 `(chunk_id, text)`。"""
    if not isinstance(record, Mapping):
        raise TypeError(f"每条记录必须是 mapping，收到 {type(record).__name__}")

    expected = {"chunk_id", "text"}
    keys = set(record)
    unknown = keys - expected
    if unknown:
        raise ValueError(
            f"记录里有不认识的键 {sorted(unknown)}，本模块只接受 {sorted(expected)}"
        )
    for name in sorted(expected):
        if name not in keys:
            raise ValueError(f"记录缺必需的键 {name!r}")

    chunk_id, text = record["chunk_id"], record["text"]
    if not isinstance(chunk_id, str):
        raise TypeError(f"chunk_id 必须是 str，收到 {type(chunk_id).__name__}")
    if not isinstance(text, str):
        raise TypeError(f"text 必须是 str，收到 {type(text).__name__}")
    return chunk_id, text


def prepare_rows(records: Iterable[Mapping]) -> list[tuple[str, str]]:
    """批量校验，返回 `[(chunk_id, text), ...]`。

    **批内 `chunk_id` 重复在这里就拦掉。** rowid 由 `chunk_id` 派生，批内重复必然
    撞 FTS5 的 rowid 主键；提前抛比让库抛清楚得多（而且能指出是哪两条）。
    """
    rows: list[tuple[str, str]] = []
    seen: dict[str, int] = {}
    for index, record in enumerate(records):
        chunk_id, text = check_record(record)
        if chunk_id in seen:
            raise ValueError(
                f"批内 chunk_id 重复：{chunk_id!r} 出现在第 {seen[chunk_id]} 条"
                f"和第 {index} 条"
            )
        seen[chunk_id] = index
        rows.append((chunk_id, text))
    return rows
