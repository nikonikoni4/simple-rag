"""建连接 —— 加载 sqlite-vec 扩展。

连接由 `VecDB` 实例持有，**不是模块级全局**：import 本模块不建立任何连接，
要几个开几个。关闭就是 `sqlite3.Connection.close()`，本身幂等（实测探针 21），
所以不额外包一层。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec


def open_connection(db_path: str | Path) -> sqlite3.Connection:
    """打开一个加载好 sqlite-vec 的连接。

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
