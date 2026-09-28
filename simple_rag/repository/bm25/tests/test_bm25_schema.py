"""`_schema` 的纯单测 —— 只有一个用例真建表，其余只构造 / 检查 DDL 文本。"""

from __future__ import annotations

import sqlite3

import pytest

from simple_rag.repository.bm25._schema import (
    CHUNK_ID_COLUMN,
    FTS_TABLE_NAME,
    FTS_TOKENIZER,
    TOKENS_COLUMN,
    build_ddl,
    verify_ddl,
)


def test_ddl_含全部关键片段():
    ddl = build_ddl()
    assert FTS_TABLE_NAME in ddl
    assert "using fts5" in ddl
    assert TOKENS_COLUMN in ddl
    assert f"{CHUNK_ID_COLUMN} unindexed" in ddl
    assert FTS_TOKENIZER in ddl


def test_ddl_能真的建出来():
    """光看字符串不算数 —— 让 SQLite 自己承认。"""
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(build_ddl())
        cols = [row[1] for row in conn.execute(f"pragma table_info({FTS_TABLE_NAME})")]
    finally:
        conn.close()
    assert cols == [TOKENS_COLUMN, CHUNK_ID_COLUMN]


def test_verify_自家_ddl_通过():
    verify_ddl(build_ddl())


def test_verify_大小写与空白不敏感():
    verify_ddl(
        "CREATE VIRTUAL TABLE chunks_fts USING FTS5("
        "tokens,   chunk_id\n UNINDEXED, tokenize='unicode61')"
    )


def test_verify_普通表报错():
    with pytest.raises(ValueError, match="不是 FTS5"):
        verify_ddl(f"create table {FTS_TABLE_NAME} (a text, b text)")


def test_verify_别的_tokenizer_报错():
    """tokenizer 不同 = token 不同 = **静默零召回** —— 必须拦。"""
    with pytest.raises(ValueError, match="tokenizer"):
        verify_ddl(
            "create virtual table chunks_fts using fts5("
            "tokens, chunk_id unindexed, tokenize='trigram')"
        )


def test_verify_缺_chunk_id_列报错():
    with pytest.raises(ValueError, match="chunk_id"):
        verify_ddl(
            "create virtual table chunks_fts using fts5("
            "tokens, tokenize='unicode61')"
        )


def test_verify_缺_tokens_列报错():
    with pytest.raises(ValueError, match="tokens"):
        verify_ddl(
            "create virtual table chunks_fts using fts5("
            "body, chunk_id unindexed, tokenize='unicode61')"
        )


def test_verify_chunk_id_没标_unindexed_报错():
    with pytest.raises(ValueError, match="unindexed"):
        verify_ddl(
            "create virtual table chunks_fts using fts5("
            "tokens, chunk_id, tokenize='unicode61')"
        )
