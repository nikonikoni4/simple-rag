"""FTS5 专属 `_store` 的纯单测 —— 只构造 SQL 文本，不碰数据库。

通用的 `check_k` / `check_record` / `prepare_rows` 已上移 `..base.validation`，
它们的测试在 `../../tests/test_validation.py`。
"""

from __future__ import annotations

import pytest

from simple_rag.repository.bm25.fts5_bm25._store import (
    ROWID_HEX_CHARS,
    build_ddl_drop,
    build_delete_by_rowids,
    build_insert,
    build_match_query,
    build_search,
    rowid_from_chunk_id,
)

# 32 位 hex —— 和本项目 Chunk.chunk_id 的形状一致
CID = "0123456789abcdef0123456789abcdef"


# ---------------------------------------------------------------- rowid


def test_rowid_取前15位hex():
    assert rowid_from_chunk_id(CID) == int(CID[:ROWID_HEX_CHARS], 16)


def test_rowid_大写hex也行():
    assert rowid_from_chunk_id(CID.upper()) == int(CID[:ROWID_HEX_CHARS], 16)


def test_rowid_只差第16位的两个_id_会撞():
    """60 bit 是**截断** —— 已知限制，写成断言别让它悄悄存在。"""
    a = "0123456789abcde" + "0" + "f" * 16
    b = "0123456789abcde" + "1" + "f" * 16
    assert a != b
    assert rowid_from_chunk_id(a) == rowid_from_chunk_id(b)


def test_rowid_刚好15位可以():
    assert rowid_from_chunk_id("f" * ROWID_HEX_CHARS) == int("f" * 15, 16)


@pytest.mark.parametrize("bad", ["c-1", "abc", "f" * 14, "xyz" + "a" * 15])
def test_rowid_非hex或太短报错(bad):
    """「是 hex 但不足 15 位」不能放过去 —— rowid 空间变小、碰撞概率上升。"""
    with pytest.raises(ValueError, match="hex"):
        rowid_from_chunk_id(bad)


def test_rowid_非_str_报错():
    with pytest.raises(TypeError):
        rowid_from_chunk_id(123)


# ---------------------------------------------------------------- match 串


def test_match_单个词加引号():
    assert build_match_query(["检索"]) == '"检索"'


def test_match_多个词用_OR():
    assert build_match_query(["a", "b"]) == '"a" OR "b"'


def test_match_内部双引号转义():
    assert build_match_query(['he"llo']) == '"he""llo"'


def test_match_空列表():
    assert build_match_query([]) == ""


def test_match_特殊字符被引号包住():
    """不加引号时 `-` / `:` / `*` 会被当成 FTS5 查询语法。"""
    assert build_match_query(["a-b", "c:d", "e*"]) == '"a-b" OR "c:d" OR "e*"'


# ---------------------------------------------------------------- SQL 文本


def test_build_insert_显式给_rowid():
    """不给 rowid 的话 FTS5 会自增，那就对不上 chunk_id 了。"""
    sql = build_insert()
    assert "rowid" in sql
    assert sql.count("?") == 3


def test_build_drop_带_if_exists():
    """rebuild 在表不在时也要能走 —— 删表必须是 if exists。"""
    sql = build_ddl_drop()
    assert "drop table if exists" in sql.lower()


def test_build_delete_占位符数量():
    assert build_delete_by_rowids(1).count("?") == 1
    assert build_delete_by_rowids(3).count("?") == 3


def test_build_delete_零行报错():
    """空 `IN ()` 是 SQL 语法错误。"""
    with pytest.raises(ValueError, match="至少删一行"):
        build_delete_by_rowids(0)


def test_build_search_按分数升序():
    """`bm25()` 返回负数，升序才是「最相关在前」。"""
    sql = build_search()
    assert "bm25(" in sql
    assert "order by score" in sql
    assert "desc" not in sql.lower()
