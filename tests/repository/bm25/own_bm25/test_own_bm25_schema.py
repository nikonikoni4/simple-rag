"""own_bm25 的表结构与 `open` 语义。

`open` 是**每次开库都会调**的一步，所以两件事都要成立：第二次不能报
`table already exists`，遇到结构不对的表也不能蒙混过去（那会静默换一套打分）。
"""

from __future__ import annotations

import sqlite3

import pytest

from simple_rag.db import Database
from simple_rag.repository.bm25 import create_bm25
from simple_rag.repository.bm25.own_bm25._schema import (
    DOCS_TABLE,
    POSTINGS_DOC_INDEX,
    POSTINGS_TABLE,
    QUERY_TABLE,
    SEEN_TABLE,
    STATS_TABLE,
    build_ddl,
    build_temp_ddl,
)


def test_ddl_能真的建出来(conn):
    """光看字符串不算数 —— 让 SQLite 自己承认，并核对列与 WITHOUT ROWID。"""
    for statement in build_ddl():
        conn.execute(statement)
    names = {
        row[0]
        for row in conn.execute("select name from sqlite_master where type = 'table'")
    }
    assert {DOCS_TABLE, POSTINGS_TABLE, STATS_TABLE} <= names
    assert conn.execute(f"pragma table_info({POSTINGS_TABLE})").fetchall()
    # WITHOUT ROWID 的 postings 没有 rowid 列
    with pytest.raises(sqlite3.OperationalError):
        conn.execute(f"select rowid from {POSTINGS_TABLE}")
    assert conn.execute(f"select n, total_dl from {STATS_TABLE}").fetchone() == (0, 0)


def test_temp_ddl_幂等(conn):
    for _ in range(2):
        for statement in build_temp_ddl():
            conn.execute(statement)
    temp_names = {
        row[0]
        for row in conn.execute("select name from sqlite_temp_master where type = 'table'")
    }
    assert {SEEN_TABLE, QUERY_TABLE} <= temp_names


def test_open_首次建表且幂等(conn, tok):
    index = create_bm25("own_bm25", tokenizer=tok, conn=conn)
    index.open()
    index.open()  # 第二次走校验分支，不能报 table already exists
    names = {
        row[0]
        for row in conn.execute("select name from sqlite_master where type = 'table'")
    }
    assert {DOCS_TABLE, POSTINGS_TABLE, STATS_TABLE} <= names


def test_temp_表不进主库文件(conn, tok):
    """TEMP 工作表在连接的 temp 库里，主库文件不该被它们污染。"""
    create_bm25("own_bm25", tokenizer=tok, conn=conn).open()
    main = {
        row[0]
        for row in conn.execute("select name from sqlite_master where name like 'own_bm25%'")
    }
    assert SEEN_TABLE not in main and QUERY_TABLE not in main
    assert POSTINGS_DOC_INDEX in main


def test_部分存在报错(conn, tok):
    """只掉了一张表 —— 补建会让两套定义混在一起，必须拦住。"""
    index = create_bm25("own_bm25", tokenizer=tok, conn=conn)
    index.open()
    conn.execute(f"drop table {STATS_TABLE}")
    with pytest.raises(ValueError, match="不完整"):
        index.open()


def test_错表报错(conn, tok):
    """三张表都在，但 `docs` 不是本模块建的那张（少了唯一约束）。"""
    index = create_bm25("own_bm25", tokenizer=tok, conn=conn)
    index.open()
    conn.execute(f"drop table {DOCS_TABLE}")
    conn.execute(
        f"create table {DOCS_TABLE} ("
        "doc_id integer primary key, chunk_id text, dl integer)"
    )
    with pytest.raises(ValueError, match="不是本模块建的"):
        index.open()


def test_错表_只建了别家的同名表(conn, tok):
    """一个空库被人先占了同名表 —— 也不能当自己的用。"""
    conn.execute(
        f"create table {DOCS_TABLE} (doc_id integer primary key, chunk_id text, dl integer)"
    )
    with pytest.raises(ValueError, match="不完整"):
        create_bm25("own_bm25", tokenizer=tok, conn=conn).open()


def test_缺索引报错(conn, tok):
    index = create_bm25("own_bm25", tokenizer=tok, conn=conn)
    index.open()
    conn.execute(f"drop index {POSTINGS_DOC_INDEX}")
    with pytest.raises(ValueError, match="缺索引"):
        index.open()


def test_关闭重开保留数据(tmp_path, tok):
    path = tmp_path / "persist.db"
    db = Database(path)
    index = create_bm25("own_bm25", tokenizer=tok, conn=db.connection)
    index.open()
    index.rebuild([{"chunk_id": "a", "text": "x y"}, {"chunk_id": "b", "text": "y"}])
    db.close()

    db = Database(path)
    try:
        reopened = create_bm25("own_bm25", tokenizer=tok, conn=db.connection)
        reopened.open()
        assert [h.chunk_id for h in reopened.search("x", k=5)] == ["a"]
    finally:
        db.close()


def test_与既有_fts5_表共存(tmp_path, tok):
    """同一个库里两套表互不干扰 —— 三方实现可以指向同一个连接。"""
    db = Database(tmp_path / "both.db")
    try:
        fts5 = create_bm25("fts5", tokenizer=tok, conn=db.connection)
        fts5.open()
        own = create_bm25("own_bm25", tokenizer=tok, conn=db.connection)
        own.open()
        data = [{"chunk_id": "0123456789abcdef", "text": "x y"}]
        fts5.rebuild(data)
        own.rebuild(data)
        assert [h.chunk_id for h in fts5.search("x", k=5)] == ["0123456789abcdef"]
        assert [h.chunk_id for h in own.search("x", k=5)] == ["0123456789abcdef"]
    finally:
        db.close()
