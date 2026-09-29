"""`Database` —— 一个 `.db` 文件的**连接所有者**。

**只管连接的创建 / 持有 / 关闭。** 不认识任何表、Schema、业务概念 ——
表由 `simple_rag.repository.vec` / `simple_rag.repository.bm25` 各自建。

**谁创建谁负责关闭**：`VecDB` / `BM25Index` 只接收注入的连接,不碰它的生命周期。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec


def open_connection(db_path: str | Path) -> sqlite3.Connection:
    """打开一个加载好 sqlite-vec 的连接。

    **函数体原样搬自 `simple_rag/repository/_connection.py`** —— 它承载 4 条实测结论
    （探针 21 / 01），不顺手改。

    派生的过程中失败就先关掉半成品再抛 —— 用局部变量，否则 `connect` 自己失败时
    关连接会抛 `AttributeError` 盖掉真实错误。
    """
    conn = sqlite3.connect(db_path)
    try:
        # Python 默认 isolation_level='' 会在 DML 前**隐式** BEGIN，
        # 那时模块再发 BEGIN 会报 `cannot start a transaction within a transaction`。
        # 设成 None，事务由模块自己显式控制
        conn.isolation_level = None

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        # 必须紧跟 load() —— 实测关掉之后再 load() 报 `not authorized`
        conn.enable_load_extension(False)

        conn.execute("pragma journal_mode=WAL")
    except Exception:
        conn.close()
        raise
    return conn


class Database:
    """持有一个连接，谁创建谁负责关闭。

    用法::

        with Database("vec.db") as database:
            conn = database.connection
            vec = VecDB(conn)
            bm25 = create_bm25("fts5", tokenizer=tokenizer, conn=conn)

    两个存储共用**同一个连接**时，它们能放进同一个事务 —— 实测 `begin` / `rollback`
    对两张表同时生效（见 [TECHNICAL.md §6.2](../../explore/bm25/TECHNICAL.md)）。

    **不提供 `transaction()` 之类的上下文管理器** —— 要用事务自己发
    `savepoint` / `release`。代价是调用方要记住成对 `release`：只 `rollback to`
    不 `release` 的话，savepoint 还留在栈上，事务一直挂着不提交,而且不报错。
    """

    def __init__(self, db_path: str | Path) -> None:
        self._conn = open_connection(db_path)

    @property
    def connection(self) -> sqlite3.Connection:
        """注入给 `VecDB` / `BM25Index` 的那个连接。"""
        return self._conn

    def close(self) -> None:
        """关闭连接。重复调用是 no-op —— `sqlite3.Connection.close()` 本身幂等
        （实测探针 21），本模块不额外维护标志位。
        """
        self._conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc_info) -> bool:
        self.close()
        return False
