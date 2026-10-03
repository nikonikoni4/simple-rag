"""own_bm25 的表形状：DDL 文本与 open 期的建表 / 校验（纯函数 + 一次校验）。

**三张主库表 + 两张连接局部 TEMP 表**，没有可配置项（不像 `vec` 的 `Schema`）。
表结构由本项目自己设计，不套 FTS5 的虚拟表形态 —— 这是第三类实现存在的理由
（见 [ADR 2026-10-02](../../../../docs/ADR/2026-10-02-新增第三类BM25实现.md)）。

| 表 | 存什么 | 为什么是这个形状 |
|---|---|---|
| `own_bm25_docs` | `chunk_id → doc_id` 映射 + 本篇长度 `dl` | `dl` 是 BM25 的归一化因子，只在本篇内可定 |
| `own_bm25_postings` | 一行一个 `(term, doc_id) -> tf` | `term` 主键支持按词遍历（打分），`doc_id` 反向索引支持按篇删（增删改） |
| `own_bm25_stats` | 全库 `N` 与 `total_dl` | `avgdl = total_dl / N` 是全库统计量，写入时**算不出来**，只能原子维护 |

**不存原文、不存 token 串、不存 IDF。** `tf` 与 `dl` 是本篇内可定的全部信息，
足够重算得分 —— 这是内存不随语料线性增长的原因（`rank_bm25` 在这里失败）。

TEMP 表只在连接内存在，不落主库：`own_bm25_seen` 拦批内重复 `chunk_id`，
`own_bm25_query` 装本次查询的词与 IDF（避免把上千个 token 拼成超大 `IN`）。
"""

from __future__ import annotations

import sqlite3

DOCS_TABLE = "own_bm25_docs"
POSTINGS_TABLE = "own_bm25_postings"
POSTINGS_DOC_INDEX = "own_bm25_postings_doc"
STATS_TABLE = "own_bm25_stats"
SEEN_TABLE = "own_bm25_seen"  # TEMP
QUERY_TABLE = "own_bm25_query"  # TEMP

MAIN_TABLES = (DOCS_TABLE, POSTINGS_TABLE, STATS_TABLE)

# 各表的 DDL 逐条留成常量：open 时拿它和 sqlite_master 里的文本比对，
# 能一次覆盖列、约束与 WITHOUT ROWID（只比列名会漏掉 CHECK 与主键形状）。
_DOCS_DDL = (
    f"create table {DOCS_TABLE} ("
    "doc_id integer primary key, "
    "chunk_id text not null unique, "
    "dl integer not null check(dl >= 0)"
    ")"
)
_POSTINGS_DDL = (
    f"create table {POSTINGS_TABLE} ("
    "term text not null, "
    "doc_id integer not null, "
    "tf integer not null check(tf > 0), "
    "primary key(term, doc_id)"
    ") without rowid"
)
_STATS_DDL = (
    f"create table {STATS_TABLE} ("
    "singleton integer primary key check(singleton = 1), "
    "n integer not null check(n >= 0), "
    "total_dl integer not null check(total_dl >= 0)"
    ")"
)
_POSTINGS_DOC_INDEX_DDL = (
    f"create index {POSTINGS_DOC_INDEX} on {POSTINGS_TABLE}(doc_id, term)"
)
_STATS_INIT = f"insert into {STATS_TABLE} (singleton, n, total_dl) values (1, 0, 0)"

_EXPECTED_DDL = {
    DOCS_TABLE: _DOCS_DDL,
    POSTINGS_TABLE: _POSTINGS_DDL,
    STATS_TABLE: _STATS_DDL,
    POSTINGS_DOC_INDEX: _POSTINGS_DOC_INDEX_DDL,
}


def build_ddl() -> tuple[str, ...]:
    """首次 open 要执行的语句，按顺序。

    空语料也计入 `N` —— `stats` 的第 0 行在这里就写好，之后只 `update` 不 `insert`。
    """
    return (_DOCS_DDL, _POSTINGS_DDL, _POSTINGS_DOC_INDEX_DDL, _STATS_DDL, _STATS_INIT)


def build_temp_ddl() -> tuple[str, ...]:
    """连接局部的 TEMP 工作表。

    `if not exists` 在这里是对的 —— `open()` 幂等，每开一次都执行这几条。
    表在连接的 temp 库里，断开即消失，不进主库文件。
    """
    return (
        f"create temp table if not exists {SEEN_TABLE} ("
        "chunk_id text primary key"
        ") without rowid",
        f"create temp table if not exists {QUERY_TABLE} ("
        "term text primary key, idf real, qtf integer"
        ") without rowid",
    )


def _normalize(sql: str) -> str:
    """大小写与空白归一 —— SQLite 存的是我们给的原文本，但比对不该挑格式。"""
    return " ".join(sql.lower().split())


def ensure_schema(conn: sqlite3.Connection) -> None:
    """建表或校验已有的表；**部分存在 / 不兼容一律报错，不自动覆盖**。

    三张表要么全是我们建的，要么全没有。只有一部分存在 = 库里有一套不完整的
    同名表（别的来源、或建到一半），这时补建会让两套定义混在一起而无人察觉。

    存在的表逐条与 `build_ddl()` 的文本比对：**只有文本级比对才覆盖
    `WITHOUT ROWID`、`CHECK`、唯一约束**，只比列名会静默接受一张结构不同的表。
    """
    present: dict[str, str | None] = {}
    for name in (*MAIN_TABLES, POSTINGS_DOC_INDEX):
        row = conn.execute(
            "select sql from sqlite_master where name = ? and type in ('table', 'index')",
            (name,),
        ).fetchone()
        present[name] = row[0] if row else None

    existing = [name for name in MAIN_TABLES if present[name] is not None]
    if not existing:
        for statement in build_ddl():
            conn.execute(statement)
        return

    missing = [name for name in MAIN_TABLES if present[name] is None]
    if missing:
        raise ValueError(
            f"{DOCS_TABLE} 这一组表不完整：缺 {missing}，已有 {existing}。"
            f"本模块不自动补建 —— 补出来的表可能和缺失的那张不是同一套定义"
        )

    for name in (*MAIN_TABLES, POSTINGS_DOC_INDEX):
        actual = present[name]
        if actual is None:
            raise ValueError(
                f"缺索引 {name!r} —— 它是按文档删词频的唯一访问路径"
                f"（缺了只能全表扫 postings）"
            )
        if _normalize(actual) != _normalize(_EXPECTED_DDL[name]):
            raise ValueError(
                f"库里的 {name!r} 不是本模块建的：{actual!r}，"
                f"本模块期望 {_EXPECTED_DDL[name]!r}"
            )


def ensure_temp_tables(conn: sqlite3.Connection) -> None:
    """建连接局部的 TEMP 工作表（幂等）。"""
    for statement in build_temp_ddl():
        conn.execute(statement)
