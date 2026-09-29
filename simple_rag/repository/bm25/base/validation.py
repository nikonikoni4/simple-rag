"""记录 / `k` 的通用校验（纯函数，实现无关）。

从原 `_store.py` 上移 —— 那里混着「所有实现都要的形状校验」和「FTS5 专属的
rowid / SQL 构造」。前者属于 base，后者留在 `fts5_bm25/_store.py`。

**chunk_id 的 hex 要求不在这里** —— 那是 FTS5 从 chunk_id 派生 rowid 的前提，
rank_bm25 实现的 chunk_id 只是普通 key，没有这个约束（差异见包 README）。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

# 实测边界（探针 18）：4097 -> `k value in knn query too large ... the limit is 4096`。
# 这条边界来自 vec0 的 knn，但两个 bm25 实现沿用同一上限 —— 统一接口的 k
# 必须有界，且不该比 vec 一路宽（调用方拿同一个小参数喂两路）。
K_MIN, K_MAX = 0, 4096


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

    **批内 `chunk_id` 重复在这里就拦掉。** 两个实现都不允许批内重复
    （fts5 撞 rowid 主键；rank_bm25 的 chunk_id → 下标映射会静默错位），
    提前抛比让实现各自报清楚得多（而且能指出是哪两条）。
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
