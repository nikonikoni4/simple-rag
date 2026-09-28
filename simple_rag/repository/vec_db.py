"""`VecDB` —— 向量存储的唯一对外类。

持有连接、Schema 与事务控制。**它不认识任何业务概念。**

两条查询路径不能互相替代（spec §2）：

- `search` —— 向量相似度查询，要查询向量、要 `k`，返回按距离排序
- `get` —— 普通查询，按 `chunk_id` 取一行，没有 `k` 也没有距离这个概念
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import _store
from ._codec import vector_to_sqlite_vector
from ._connection import open_connection
from ._schema import TABLE_NAME, Schema, build_ddl, parse_ddl


@dataclass(frozen=True)
class VecSearchResult:
    """`search` 与 `get` 的统一返回 —— 不新增第二个类型。

    **不返回向量**（见 spec §12）。
    """

    chunk_id: str
    distance: float | None  # search 时是距离；get 时是 None
    created_at: str
    updated_at: str
    fields: dict[str, str | None]


def _now() -> str:
    """ISO 8601 + UTC（spec §3.1）。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class VecDB:
    """一个 `.db` 文件一个实例。

    **不加 `check_same_thread=False`** —— 保持 Python 默认的线程亲和性，
    一个实例不跨线程使用。
    """

    def __init__(self, db_path: str | Path) -> None:
        self._conn = open_connection(db_path)
        self._schema: Schema | None = None

    def __enter__(self) -> "VecDB":
        return self

    def __exit__(self, *exc_info) -> bool:
        self.close()
        return False

    # ---------------------------------------------------------------- 生命周期

    def close(self) -> None:
        """关闭连接。重复调用是 no-op。

        关闭后再调用其他方法**不做额外检查** —— sqlite3 自己会抛
        `ProgrammingError`（实测探针 21）。
        """
        self._conn.close()

    def create_table(self, schema: Schema) -> None:
        """建表；表已存在则比对 `dim` 与 `metric`。

        每次开库都调它是最常规的用法，所以第二次打开同一个 `.db` 不能报
        `table already exists`；也不能用 `if not exists` 蒙混过去 ——
        那会静默接受一个结构不同的库。

        只比 dim 与 metric：全 DDL 比对会在模块自己的生成器改动时误报，
        而这两个是**唯一会静默出错**的参数（其余字段不符会在 insert / search
        时明确报错）。
        """
        existing = self._read_ddl()
        if existing is None:
            self._conn.execute(build_ddl(schema))
        else:
            dim, metric = parse_ddl(existing)
            if dim != schema.dim or metric.lower() != schema.metric.lower():
                raise ValueError(
                    f"打开的不是这个 Schema 建的库：库里是 dim={dim} metric={metric}，"
                    f"传入的是 dim={schema.dim} metric={schema.metric}"
                )
        self._schema = schema

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
        """整批插入，**一个事务**：全部成功或全部失败。

        `chunk_id` 冲突无法提前查（批内重复只有库知道），失败发生在事务中途，
        这时**显式 `rollback`** —— 实测语句失败不会自动中止事务，退化成半批写入。
        """
        schema = self._require_schema()

        rows: list[tuple] = []
        timestamp = _now()
        # 边遍历边materialize，生成器只会被消费一次
        for record in records:
            chunk_id = _store.check_record(schema, record)
            blob = vector_to_sqlite_vector(record["vector"], schema.dim)
            rows.append(
                (
                    chunk_id,
                    blob,
                    timestamp,
                    timestamp,
                    *(record[name] for name in schema.fields),
                )
            )
        if not rows:
            return

        conn = self._conn
        conn.execute("begin")
        try:
            conn.executemany(_store.build_insert(schema), rows)
        except sqlite3.OperationalError as exc:
            conn.rollback()
            # 实测（探针 18 §8.4）：vec0 抛的是 OperationalError 而不是
            # IntegrityError，sqlite_errorcode 还是通用的 1，只能匹配文本。
            # 表上只有主键一个唯一约束，所以匹配到这里就是 chunk_id 冲突
            if "UNIQUE constraint failed" in str(exc):
                raise ValueError(f"chunk_id 重复：{exc}") from exc
            raise
        except BaseException:
            conn.rollback()
            raise
        conn.execute("commit")

    def update(
        self,
        chunk_id: str,
        vector=None,
        fields: Mapping | None = None,
    ) -> int:
        """局部更新：`fields` 里给谁改谁。返回受影响行数，`0` 表示 chunk_id 不存在。

        **直接改列，不是「先删再插」** —— 实测 vec0 支持改向量列和各列（含加号列），
        走 delete + insert 会重置 `created_at`。
        """
        schema = self._require_schema()
        if not isinstance(chunk_id, str):
            raise TypeError(f"chunk_id 必须是 str，收到 {type(chunk_id).__name__}")
        if fields is not None and not isinstance(fields, Mapping):
            raise TypeError(f"fields 必须是 mapping，收到 {type(fields).__name__}")

        changes = dict(fields) if fields else {}
        if vector is None and not changes:
            raise ValueError("update 至少要给 vector 或 fields 之一")

        # columns 与 params 同步追加，参数顺序自然对上
        columns: list[str] = ["updated_at"]
        params: list = [_now()]
        if vector is not None:
            columns.append("embedding")
            params.append(vector_to_sqlite_vector(vector, schema.dim))
        for name, value in changes.items():
            # chunk_id / created_at / updated_at 都不在 schema.fields 里，
            # 于是「绝不生成 SET chunk_id = ...」是自然成立的
            if name not in schema.fields:
                raise ValueError(
                    f"{name!r} 不是已声明的字段，不能更新；"
                    f"已声明的是 {list(schema.fields)}"
                )
            if value is not None and not isinstance(value, str):
                raise TypeError(
                    f"字段 {name!r} 的值必须是 str 或 None，收到 {type(value).__name__}"
                )
            columns.append(name)
            params.append(value)
        params.append(chunk_id)

        cursor = self._conn.execute(_store.build_update(columns), params)
        return cursor.rowcount

    def delete(self, chunk_id: str) -> int:
        """按 id 删一行，返回删除行数（不存在返回 `0`）。"""
        self._require_schema()
        if not isinstance(chunk_id, str):
            raise TypeError(f"chunk_id 必须是 str，收到 {type(chunk_id).__name__}")
        cursor = self._conn.execute(_store.build_delete(), (chunk_id,))
        return cursor.rowcount

    # ---------------------------------------------------------------- 读

    def search(self, vector, k: int, where=None) -> list[VecSearchResult]:
        """向量相似度查询：最近的 `k` 条，可加等值过滤。

        - 用 `k = ?` 约束，不用 `order by distance limit`
        - 距离并列时的顺序**不是稳定契约**；返回条数可能少于 `k`
        - `where` 只能过滤 `chunk_id` 与显式声明为可过滤的字段
        """
        schema = self._require_schema()
        k = _store.check_k(k)
        conditions = _store.check_where(schema, where)
        blob = vector_to_sqlite_vector(vector, schema.dim)

        sql, keys = _store.build_search(schema, conditions)
        params = [blob, *(conditions[key] for key in keys), k]
        rows = self._conn.execute(sql, params).fetchall()
        return [self._to_result(schema, row, from_search=True) for row in rows]

    def get(self, chunk_id: str) -> VecSearchResult | None:
        """按 id 取一行，不存在返回 `None`。

        **不受 `k`、不受距离、不受列类型限制** —— 普通查询，两类列都能取。
        没有过滤参数（要过滤用 `search` 的 `where`）。
        """
        schema = self._require_schema()
        if not isinstance(chunk_id, str):
            raise TypeError(f"chunk_id 必须是 str，收到 {type(chunk_id).__name__}")
        row = self._conn.execute(_store.build_get(schema), (chunk_id,)).fetchone()
        if row is None:
            return None
        return self._to_result(schema, row, from_search=False)

    @staticmethod
    def _to_result(
        schema: Schema, row: tuple, *, from_search: bool
    ) -> VecSearchResult:
        width = len(schema.fields)
        return VecSearchResult(
            chunk_id=row[0],
            distance=row[3 + width] if from_search else None,
            created_at=row[1],
            updated_at=row[2],
            fields=dict(zip(schema.fields, row[3 : 3 + width])),
        )
