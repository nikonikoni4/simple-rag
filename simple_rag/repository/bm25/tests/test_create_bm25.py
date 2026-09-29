"""`create_bm25` 工厂的测试 —— 参数路由与错配显式报错。"""

from __future__ import annotations

import pytest

from simple_rag.db import Database
from simple_rag.repository.bm25 import BM25Index, create_bm25
from simple_rag.repository.bm25.fts5_bm25 import FTS5BM25Index
from simple_rag.repository.bm25.rank_bm25 import RankBM25Index
from simple_rag.tokenization import TokenizerFactory


@pytest.fixture
def tok():
    return TokenizerFactory.create("jieba")


@pytest.fixture
def database(tmp_path):
    db = Database(tmp_path / "bm25.db")
    yield db
    db.close()


# ---------------------------------------------------------------- 路由


def test_fts5_路由到_FTS5BM25Index(database, tok):
    index = create_bm25("fts5", tokenizer=tok, conn=database.connection)
    assert isinstance(index, FTS5BM25Index)
    assert isinstance(index, BM25Index)


def test_rank_bm25_路由到_RankBM25Index(tok):
    index = create_bm25("rank_bm25", tokenizer=tok)
    assert isinstance(index, RankBM25Index)
    assert isinstance(index, BM25Index)


def test_未知实现报错(tok):
    with pytest.raises(ValueError, match="未知的 bm25 实现"):
        create_bm25("elasticsearch", tokenizer=tok)


# ---------------------------------------------------------------- 参数错配


def test_fts5_缺_conn_报错(tok):
    with pytest.raises(ValueError, match="conn"):
        create_bm25("fts5", tokenizer=tok)


def test_fts5_传_k1_报错(database, tok):
    with pytest.raises(ValueError, match="k1"):
        create_bm25("fts5", tokenizer=tok, conn=database.connection, k1=1.2)


def test_fts5_显式传默认k1_也报错(database, tok):
    """None 是哨兵 —— 显式传 1.5 和没传不可区分时放行，调用方会以为参数生效了
    （FTS5 实际硬编码 1.2）。"""
    with pytest.raises(ValueError, match="k1"):
        create_bm25("fts5", tokenizer=tok, conn=database.connection, k1=1.5)
    with pytest.raises(ValueError, match="b="):
        create_bm25("fts5", tokenizer=tok, conn=database.connection, b=0.75)


def test_fts5_传_b_报错(database, tok):
    with pytest.raises(ValueError, match="b="):
        create_bm25("fts5", tokenizer=tok, conn=database.connection, b=0.6)


def test_fts5_传_persist_path_报错(database, tok):
    with pytest.raises(ValueError, match="persist_path"):
        create_bm25(
            "fts5", tokenizer=tok, conn=database.connection, persist_path="x.pkl"
        )


def test_rank_bm25_传_conn_报错(database, tok):
    with pytest.raises(ValueError, match="conn"):
        create_bm25("rank_bm25", tokenizer=tok, conn=database.connection)
