"""own_bm25 的数据访问层 —— 参数化读写与打分 SQL（**不接触分词器**）。

分词、事务编排、批内重复的判据都在 `index.py`；这里只负责「一条 SQL 做什么」。
分开的理由是 SQL 的形状（表名、列名、打分表达式）集中在一处，测试能直接对
某条语句核对，而 `index.py` 保持成流程。

**全部走参数绑定。** token 与 `chunk_id` 都是调用方给的字符串，任何一处拼进
SQL 文本都是注入面；打分表达式里的 `k1` / `b` / `avgdl` 也一律绑参数。

**打分 SQL 用代数等价式而不是字面公式**：字面写法
`tf*(k1+1)/(tf + k1*X)` 在有限但极大的 `k1`（如 1e308）下 `tf*(k1+1)` 溢出成
`inf`，整个候选集返回 `inf`/`nan`。分母分子同除 `(k1+1)` 后只剩
`tf/(tf/(k1+1) + (k1/(k1+1))*X)`，`k1→∞` 时 `tf/(k1+1) → 0`、`k1/(k1+1) → 1`，
结果有限且等于极限值（实测 16 组极端参数均为有限负分）。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

from ._schema import (
    DOCS_TABLE,
    POSTINGS_TABLE,
    QUERY_TABLE,
    SEEN_TABLE,
    STATS_TABLE,
)

# `k1_plus_one = k1 + 1`、`k1_ratio = k1 / (k1 + 1)` 由 Python 预算好再绑参 ——
# 放进 SQL 里会让 SQLite 在每个候选行上重算一遍。
SEARCH_SQL = f"""
select d.chunk_id,
       -sum(q.idf * q.qtf *
            (p.tf / (p.tf / :k1_plus_one +
             :k1_ratio * (1.0 - :b + :b * d.dl / :avgdl)))) as score
from temp.{QUERY_TABLE} as q
cross join {POSTINGS_TABLE} as p
cross join {DOCS_TABLE} as d
where p.term = q.term and d.doc_id = p.doc_id
group by d.doc_id
order by score asc, d.chunk_id collate binary asc
limit :k
"""


# ---------------------------------------------------------------- 全库统计

# 统计行只有 `open` 建表时插一次，之后只 update 不 insert。它不见了只可能是
# 「绕过接口改了索引表」—— 读写两条路径都**显式报错**，不退化成静默的错误结果：
# 读的时候静默 → 空结果，看着像「没有命中」；写的时候静默 → 返回成功计数，
# 但统计量已经不更新了。
_MISSING_STATS = f"{STATS_TABLE} 缺少 singleton = 1 的行 —— 索引被绕过接口改过"


def _check_stats_written(cursor) -> None:
    if cursor.rowcount != 1:
        raise ValueError(f"{_MISSING_STATS}，写了但统计量没更新")


def read_stats(conn) -> tuple[int, int]:
    """读 `(N, total_dl)`。"""
    row = conn.execute(
        f"select n, total_dl from {STATS_TABLE} where singleton = 1"
    ).fetchone()
    if row is None:
        raise ValueError(f"{_MISSING_STATS}，无法查询")
    return int(row[0]), int(row[1])


def bump_stats(conn, n_delta: int, dl_delta: int) -> None:
    """按增量原子更新 `N` 与总长度。整批写完后调一次。"""
    _check_stats_written(
        conn.execute(
            f"update {STATS_TABLE} set n = n + ?, total_dl = total_dl + ? "
            f"where singleton = 1",
            (n_delta, dl_delta),
        )
    )


def reset_stats(conn) -> None:
    _check_stats_written(
        conn.execute(
            f"update {STATS_TABLE} set n = 0, total_dl = 0 where singleton = 1"
        )
    )


# ---------------------------------------------------------------- 文档与词频


def find_doc(conn, chunk_id: str) -> tuple[int, int] | None:
    """`chunk_id` → `(doc_id, dl)`；不存在返回 None。"""
    return conn.execute(
        f"select doc_id, dl from {DOCS_TABLE} where chunk_id = ?", (chunk_id,)
    ).fetchone()


def insert_doc(conn, chunk_id: str, tokens: Sequence[str]) -> int:
    """写一篇：`docs` 一行 + `postings` 每个不同 token 一行。返回本篇长度 `dl`。

    `dl` 是**分词结果的总长度（含重复 token）**，不是不同词数 —— BM25 的
    长度归一化用的是前者，用后者会静默换一套打分。
    """
    dl = len(tokens)
    cursor = conn.execute(
        f"insert into {DOCS_TABLE} (chunk_id, dl) values (?, ?)", (chunk_id, dl)
    )
    doc_id = cursor.lastrowid
    counts = Counter(tokens)
    if counts:
        conn.executemany(
            f"insert into {POSTINGS_TABLE} (term, doc_id, tf) values (?, ?, ?)",
            ((term, doc_id, tf) for term, tf in counts.items()),
        )
    return dl


def delete_doc(conn, doc_id: int) -> None:
    """删一篇：先按 `doc_id` 反向索引删词频，再删映射行。

    postings 的 `(doc_id, term)` 索引就是为这条语句建的 —— 没有它只能全表扫。
    """
    conn.execute(f"delete from {POSTINGS_TABLE} where doc_id = ?", (doc_id,))
    conn.execute(f"delete from {DOCS_TABLE} where doc_id = ?", (doc_id,))


def clear_index(conn) -> None:
    """清空语料并重置统计 —— `rebuild` 的第一步。"""
    conn.execute(f"delete from {POSTINGS_TABLE}")
    conn.execute(f"delete from {DOCS_TABLE}")
    reset_stats(conn)


# ---------------------------------------------------------------- 批内去重（TEMP）


def clear_seen(conn) -> None:
    conn.execute(f"delete from temp.{SEEN_TABLE}")


def mark_seen(conn, chunk_id: str) -> bool:
    """首次见到返回 True，批内重复返回 False。

    用 `insert or ignore` + `rowcount` 判重，**不建与语料规模同步增长的 Python
    `set`** —— 10 万篇时那是几十 MB 的常驻对象，而这张 TEMP 表不进主库文件。
    """
    cursor = conn.execute(
        f"insert or ignore into temp.{SEEN_TABLE} (chunk_id) values (?)", (chunk_id,)
    )
    return cursor.rowcount == 1


# ---------------------------------------------------------------- 查询期（TEMP）


def count_df(conn, term: str) -> int:
    """`df(t)` —— 主键 `(term, doc_id)` 上的范围计数，不是全表扫。"""
    row = conn.execute(
        f"select count(*) from {POSTINGS_TABLE} where term = ?", (term,)
    ).fetchone()
    return int(row[0])


def clear_query(conn) -> None:
    conn.execute(f"delete from temp.{QUERY_TABLE}")


def fill_query(conn, rows: Iterable[tuple[str, float, int]]) -> None:
    """把 `(term, idf, qtf)` 写进 TEMP 表，供打分 SQL 连接。

    这一步替掉「把上千个 token 拼成 `IN (...)`」—— 参数个数不再受
    `SQLITE_MAX_VARIABLE_NUMBER` 限制。
    """
    conn.executemany(
        f"insert into temp.{QUERY_TABLE} (term, idf, qtf) values (?, ?, ?)", rows
    )


def run_search(
    conn, *, k1_plus_one: float, k1_ratio: float, b: float, avgdl: float, k: int
) -> list[tuple[str, float]]:
    """打分 + 聚合 + 排序都在 SQLite 里做完，只取最终 `k` 行。"""
    return conn.execute(
        SEARCH_SQL,
        {
            "k1_plus_one": k1_plus_one,
            "k1_ratio": k1_ratio,
            "b": b,
            "avgdl": avgdl,
            "k": k,
        },
    ).fetchall()
