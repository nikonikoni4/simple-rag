"""`FTS5BM25Index` —— 基于 SQLite FTS5 的 BM25 实现。

**与 `VecDB` 互不依赖**，两者只共用调用方注入的连接与分词器。实例经
`create_bm25("fts5", ...)` 获得，本类**不在包出口里** —— 统一接口见 `..base`。

存进去的是**分词后的 token 串**（由构造时注入的 `Tokenizer` 切），因为 FTS5 的
`unicode61` 对中文无效 —— 直接喂原文会整句一个 token，查 `检索` 零命中
（实测 [explore/bm25/FINDINGS.md](../../../../explore/bm25/FINDINGS.md) §4.1）。
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable, Mapping

from simple_rag.tokenization import Tokenizer, to_document

from ..base import BM25Index, BM25SearchResult, check_k, prepare_rows
from . import _store
from ._schema import FTS_TABLE_NAME, build_ddl, verify_ddl

logger = logging.getLogger(__name__)

_INSERT_SAVEPOINT = "fts5_insert"
_REBUILD_SAVEPOINT = "fts5_rebuild"
_REPLACE_SAVEPOINT = "fts5_replace"


def _rowid_conflict(exc: sqlite3.IntegrityError) -> ValueError:
    """翻译 FTS5 的 rowid 撞车 —— `insert` / `replace` / `rebuild` 共用。

    实测：撞 rowid 时 FTS5 抛 `IntegrityError("constraint failed")`，裸消息
    看不出原因。两种来路 —— 库里已有同 `chunk_id`，或两个 `chunk_id` 的前
    15 位 hex 相同（rowid 是 60 bit 截断，见 `_store.rowid_from_chunk_id`）。
    三个写方法都必须翻译成同一个异常类型，调用方才不用分别接。
    """
    return ValueError(
        f"rowid 冲突（{exc}）：可能是这个 chunk_id 已存在，或两个 chunk_id "
        f"的前 {_store.ROWID_HEX_CHARS} 位 hex 相同"
    )


class FTS5BM25Index(BM25Index):
    """FTS5 关键词索引。

    **不持有连接** —— 由调用方注入（`simple_rag.db.Database`），本类不创建、
    也不关闭。**没有 `close()`**：留一个什么都不做的 `close()` 更危险 ——
    调用方以为收尾了，实际连接还开着。

    `k1` / `b` 由 FTS5 硬编码在 1.2 / 0.75（见已知限制文档），本实现不可调；
    要调参用 `rank_bm25` 实现。
    """

    def __init__(self, conn: sqlite3.Connection, tokenizer: Tokenizer) -> None:
        self._conn = conn
        self._tokenizer = tokenizer

    # ---------------------------------------------------------------- 打开

    def open(self) -> None:
        """幂等地把索引恢复到可查询：表不存在则建，存在则比对 DDL。

        每次开库都调它是最常规的用法，所以第二次打开同一个 `.db` 不能报
        `table already exists`；也不能用 `if not exists` 蒙混过去 —— 那会静默接受
        一个 tokenizer 不同的表（token 不同 = **静默零召回**）。
        """
        existing = self._read_ddl()
        if existing is None:
            self._conn.execute(build_ddl())
        else:
            verify_ddl(existing)

    def _read_ddl(self) -> str | None:
        row = self._conn.execute(
            "select sql from sqlite_master where name = ? and type = 'table'",
            (FTS_TABLE_NAME,),
        ).fetchone()
        return row[0] if row else None

    # ---------------------------------------------------------------- 写

    def rebuild(self, records: Iterable[Mapping]) -> int:
        """全量重建：一个事务内删表 + 建表 + 全量插入，返回写入条数。

        原有数据全部丢弃 —— 换分词器后的标准恢复路径就是它。空批也把表清掉：
        「重建为空」和「重建为这批」语义一致，调用方不该靠传不传空批区分。
        """
        rows = prepare_rows(records)
        params = [self._to_params(chunk_id, text) for chunk_id, text in rows]
        conn = self._conn
        conn.execute(f"savepoint {_REBUILD_SAVEPOINT}")
        try:
            conn.execute(_store.build_ddl_drop())
            conn.execute(build_ddl())
            if params:
                conn.executemany(_store.build_insert(), params)
        except sqlite3.IntegrityError as exc:
            # 删过表了还能撞，只剩一种来路 —— 批内两个 chunk_id 前 15 位 hex 相同
            conn.execute(f"rollback to {_REBUILD_SAVEPOINT}")
            conn.execute(f"release {_REBUILD_SAVEPOINT}")
            raise _rowid_conflict(exc) from exc
        except BaseException:
            conn.execute(f"rollback to {_REBUILD_SAVEPOINT}")
            conn.execute(f"release {_REBUILD_SAVEPOINT}")
            raise
        conn.execute(f"release {_REBUILD_SAVEPOINT}")
        return len(params)

    def insert(self, records: Iterable[Mapping]) -> int:
        """整批插入，返回写入行数。

        `records` 每项 `{"chunk_id": str, "text": str}` —— `text` 是**原文**，
        分词在本类内做。非 hex / 不足 15 位的 `chunk_id`、批内重复，都在写库前抛。

        用 `SAVEPOINT` 而不是 `BEGIN` —— 调用方可能已经开着事务（要把本表和 vec0
        表的写入圈进同一个事务），那时 `BEGIN` 会报
        `cannot start a transaction within a transaction`（实测）。`SAVEPOINT`
        在事务外会自己开一个，在事务内会成为子事务，**两种用法都对**。
        """
        rows = prepare_rows(records)
        if not rows:
            return 0
        params = [self._to_params(chunk_id, text) for chunk_id, text in rows]

        conn = self._conn
        conn.execute(f"savepoint {_INSERT_SAVEPOINT}")
        try:
            conn.executemany(_store.build_insert(), params)
        except sqlite3.IntegrityError as exc:
            conn.execute(f"rollback to {_INSERT_SAVEPOINT}")
            conn.execute(f"release {_INSERT_SAVEPOINT}")
            raise _rowid_conflict(exc) from exc
        except BaseException:
            conn.execute(f"rollback to {_INSERT_SAVEPOINT}")
            conn.execute(f"release {_INSERT_SAVEPOINT}")
            raise
        conn.execute(f"release {_INSERT_SAVEPOINT}")
        return len(params)

    def replace(self, records: Iterable[Mapping]) -> int:
        """覆盖或插入：一个事务内先删同 id 旧行再插入，返回写入条数。

        `chunk_id` 已存在则覆盖，不存在也直接写入 —— 调用方不用先判断存在性。
        FTS5 表不支持原地改列，「改 = 删 + 插」在这里被一个 savepoint 圈成原子。
        """
        rows = prepare_rows(records)
        if not rows:
            return 0
        params = [self._to_params(chunk_id, text) for chunk_id, text in rows]
        rowids = [p[0] for p in params]

        conn = self._conn
        conn.execute(f"savepoint {_REPLACE_SAVEPOINT}")
        try:
            conn.execute(_store.build_delete_by_rowids(len(rowids)), rowids)
            conn.executemany(_store.build_insert(), params)
        except sqlite3.IntegrityError as exc:
            # 旧行已删，还能撞就是批内两个 chunk_id 前 15 位 hex 相同
            conn.execute(f"rollback to {_REPLACE_SAVEPOINT}")
            conn.execute(f"release {_REPLACE_SAVEPOINT}")
            raise _rowid_conflict(exc) from exc
        except BaseException:
            conn.execute(f"rollback to {_REPLACE_SAVEPOINT}")
            conn.execute(f"release {_REPLACE_SAVEPOINT}")
            raise
        conn.execute(f"release {_REPLACE_SAVEPOINT}")
        return len(params)

    def delete(self, chunk_ids: Iterable[str]) -> int:
        """按 `chunk_id` 批量删，返回删除行数。

        内部用 `rowid` 定位（`chunk_id` 是 `unindexed` 列，用它定位是**全表扫**），
        一条 SQL 删完 —— 实测比逐条按 `chunk_id` 快 **127 倍**。
        """
        ids = list(chunk_ids)
        if not ids:
            return 0
        rowids = [_store.rowid_from_chunk_id(chunk_id) for chunk_id in ids]
        cursor = self._conn.execute(
            _store.build_delete_by_rowids(len(rowids)), rowids
        )
        return cursor.rowcount

    def _to_params(self, chunk_id: str, text: str) -> tuple[int, str, str]:
        """一条记录 → insert 参数 `(rowid, token 串, chunk_id)`。"""
        return (
            _store.rowid_from_chunk_id(chunk_id),
            # 分隔符是空格 —— 分词器已保证 token 不含内部空格
            to_document(self._tokenizer.tokenize(text)),
            chunk_id,
        )

    # ---------------------------------------------------------------- 读

    def search(self, query: str, k: int) -> list[BM25SearchResult]:
        """关键词查询，按相关度排序。

        `query` 是**原文**，分词在本类内做。分不出 token 时（空串、纯空白、纯
        标点）直接返回 `[]` —— `match ''` 是 FTS5 语法错误。

        ⚠️ 分数是**负数**，越小越相关（FTS5 的 `bm25()` 整体取负）。
        ⚠️ `order by bm25()` 要给**全部命中文档**打分，所以耗时随命中集**线性增长**，
        不是 O(log N)（5 万条 63.4 ms 实测）。
        """
        k = check_k(k)
        if k == 0:
            return []
        tokens = self._tokenizer.tokenize(query)
        if not tokens:
            return []
        rows = self._conn.execute(
            _store.build_search(), (_store.build_match_query(tokens), k)
        ).fetchall()
        return [BM25SearchResult(chunk_id=row[0], score=row[1]) for row in rows]
