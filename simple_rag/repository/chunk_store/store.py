"""`ChunkStore` —— 数据表的唯一对外类。

**不持有连接** —— 连接由调用方注入（`simple_rag.db.Database`），本类不创建、
也不关闭它。**它不认识任何业务概念。**

数据表与向量表（`repository/vec`）是**并列**的两张表，分工不同：

- 向量表管「哪些 chunk 跟这个查询像」—— 只做 KNN
- 数据表管「这个 chunk 是什么内容、来自哪些文件的哪几行」

检索时的顺序固定：向量表给出 `chunk_id`，再回这里取正文与来源。
见 `simple_rag/repository/表结构约定.md`。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone

from . import _store
from ._schema import BASE_COLUMNS, TABLE_NAME, Schema, build_ddl, build_indexes


@dataclass(frozen=True)
class ChunkRow:
    """数据表的一行 —— 一个 chunk 的**一条来源**。

    Attributes:
        chunk_id: 所属 chunk。
        path: 来源文件路径。
        start_line: 起始行（闭区间）。
        end_line: 结束行（闭区间）。
        created_at / updated_at: ISO 8601 + UTC，由模块维护。
        content: 正文。**同一 chunk 的各行里是同一份**，所以多来源时重复存放。
        fields: 调用方声明的扩展列。
    """

    chunk_id: str
    path: str
    start_line: int
    end_line: int
    created_at: str
    updated_at: str
    content: str
    fields: dict[str, str | None]


def _now() -> str:
    """ISO 8601 + UTC。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ChunkStore:
    """持有一个**注入的**连接。

    与 `VecDB` 一样：**没有 `close()`** —— 连接归调用方（`simple_rag.db.Database`）
    管；**不加 `check_same_thread=False`** —— 保持 Python 默认的线程亲和性，
    一个实例不跨线程使用。
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._schema: Schema | None = None

    # ---------------------------------------------------------------- 建表

    def create_table(self, schema: Schema) -> None:
        """建表并建索引；表已存在就直接用。

        这里**不像 `VecDB` 那样比对结构**：普通表的加列本来就允许演进
        （`ALTER TABLE` 即可），比对列集合只会在调用方自己加了列之后误报。
        真对不上时插入会报 `no such column` —— 那是明确的错误，不会静默出错。
        """
        self._schema = schema
        if self._read_ddl() is None:
            self._conn.execute(build_ddl(schema))
            for statement in build_indexes():
                self._conn.execute(statement)

    def _read_ddl(self) -> str | None:
        row = self._conn.execute(
            "select sql from sqlite_master where name = ? and type = 'table'",
            (TABLE_NAME,),
        ).fetchone()
        return row[0] if row else None

    def _require_schema(self) -> Schema:
        if self._schema is None:
            raise ValueError("还没调 create_table —— Schema 未知，无法操作")
        return self._schema

    # ---------------------------------------------------------------- 写

    def insert(self, records: Iterable[Mapping]) -> None:
        """整批插入，**一个 `SAVEPOINT`**：全部成功或全部失败。

        一条记录 = **一个 chunk**（形状见 `_store.check_record`）。`sources` 有 N 项
        就往表里写 N 行 —— 同一 chunk 的 `content` 与扩展列在各行重复，换来的是
        「一条 SQL 按 `path` 定位」和「按 `chunk_id` 一次取回全部来源」。

        用 `SAVEPOINT` 而不是 `BEGIN`：调用方通常要把数据表和向量表的写入圈进
        同一个事务，那时 `BEGIN` 会报 `cannot start a transaction within a
        transaction`。失败时**必须 `rollback to` + `release`** —— 只回滚不释放
        会让事务一直挂着不提交，**而且不报错**。
        """
        schema = self._require_schema()

        rows: list[tuple] = []
        timestamp = _now()
        # 边遍历边materialize，生成器只会被消费一次
        for record in records:
            chunk_id, sources = _store.check_record(schema, record)
            extra = [record[name] for name in schema.fields]
            for source in sources:
                rows.append(
                    (
                        chunk_id,
                        source["path"],
                        source["start_line"],
                        source["end_line"],
                        timestamp,
                        timestamp,
                        record["content"],
                        *extra,
                    )
                )
        if not rows:
            return

        conn = self._conn
        conn.execute("savepoint chunk_data_insert")
        try:
            conn.executemany(_store.build_insert(schema), rows)
        except BaseException:
            conn.execute("rollback to chunk_data_insert")
            conn.execute("release chunk_data_insert")
            raise
        conn.execute("release chunk_data_insert")

    def delete(self, chunk_id: str) -> int:
        """按 id 删掉这个 chunk 的**全部行**，返回删除行数（不存在返回 `0`）。"""
        self._require_schema()
        if not isinstance(chunk_id, str):
            raise TypeError(f"chunk_id 必须是 str，收到 {type(chunk_id).__name__}")
        cursor = self._conn.execute(_store.build_delete_by_chunk_id(), (chunk_id,))
        return cursor.rowcount

    def delete_by_path(self, path: str) -> int:
        """按文件删来源行，返回删除行数。

        **只动数据表。** 一个文件被改动时数据表这侧由它清干净；向量表那侧要不要
        跟着删，由调用方决定（先 `ids_by_path` 拿到 id，再去删向量表）——
        本类不跨模块替它拿主意。
        """
        self._require_schema()
        if not isinstance(path, str):
            raise TypeError(f"path 必须是 str，收到 {type(path).__name__}")
        cursor = self._conn.execute(_store.build_delete_by_path(), (path,))
        return cursor.rowcount

    # ---------------------------------------------------------------- 读

    def get(self, chunk_id: str) -> list[ChunkRow]:
        """按 id 取回该 chunk 的全部行，**按 `path` / `start_line` 排序**。

        不存在时返回**空列表**而不是 `None` —— 「有没有」和「有几条」是同一个
        问题的两种问法，用同一种返回值表达更省事。
        """
        schema = self._require_schema()
        if not isinstance(chunk_id, str):
            raise TypeError(f"chunk_id 必须是 str，收到 {type(chunk_id).__name__}")
        rows = self._conn.execute(_store.build_get(schema), (chunk_id,)).fetchall()
        return [self._to_row(schema, row) for row in rows]

    def ids_by_path(self, path: str) -> list[str]:
        """列出所有来源于该文件的 `chunk_id`（已去重，顺序不保证）。

        更新一个文件时的第一步：先知道要动哪些 chunk，再去各自的表里处理。
        """
        self._require_schema()
        if not isinstance(path, str):
            raise TypeError(f"path 必须是 str，收到 {type(path).__name__}")
        rows = self._conn.execute(_store.build_ids_by_path(), (path,)).fetchall()
        return [row[0] for row in rows]

    @staticmethod
    def _to_row(schema: Schema, row: tuple) -> ChunkRow:
        """按 `BASE_COLUMNS` 的名字取值，**不写死下标** ——
        顺序将来变了也不会静默错位（`ChunkRow` 的字段名与它是同一批）。"""
        base = len(BASE_COLUMNS)
        return ChunkRow(
            **dict(zip(BASE_COLUMNS, row[:base])),
            fields=dict(zip(schema.fields, row[base:])),
        )
