"""`Database` 的测试 —— 真开临时库，不 mock。"""

from __future__ import annotations

import sqlite3

import pytest

from simple_rag.db import Database
from simple_rag.db.database import open_connection


def test_打开后能建_vec0_表(tmp_path):
    """扩展真的加载上了 —— 没加载的话 `using vec0(...)` 报 `no such module`。"""
    with Database(tmp_path / "t.db") as database:
        database.connection.execute(
            "create virtual table t using vec0(v float[2])"
        )


def test_连接是手动事务模式(tmp_path):
    """`isolation_level = None`。

    默认的 `''` 会在 DML 前**隐式** BEGIN，那时模块再发 `savepoint` 会撞
    `cannot start a transaction within a transaction`（实测）。
    """
    with Database(tmp_path / "t.db") as database:
        assert database.connection.isolation_level is None


def test_close_重复调用是_noop(tmp_path):
    """`sqlite3.Connection.close()` 本身幂等（实测探针 21），本模块不维护标志位。"""
    database = Database(tmp_path / "t.db")
    database.close()
    database.close()
    database.close()


def test_退出_with_后连接已关(tmp_path):
    with Database(tmp_path / "t.db") as database:
        conn = database.connection
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("select 1")


def test_关闭后再操作抛_ProgrammingError(tmp_path):
    """不做额外检查 —— sqlite3 自己会抛。"""
    database = Database(tmp_path / "t.db")
    database.close()
    with pytest.raises(sqlite3.ProgrammingError):
        database.connection.execute("select 1")


def test_不导出裸连接工厂():
    """`__all__` 只有 `Database`。

    两层公开入口会让人不知道该用哪个，也会让「谁负责关闭」重新变得模糊。
    """
    import simple_rag.db as db_pkg

    assert db_pkg.__all__ == ["Database"]
    assert not hasattr(db_pkg, "open_connection")
    assert callable(open_connection)  # 深路径仍然可用
