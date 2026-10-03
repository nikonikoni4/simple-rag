"""`create_bm25` 工厂的测试 —— 参数路由、错配报错与参数域边界。"""

from __future__ import annotations

import math

import pytest

from simple_rag.db import Database
from simple_rag.repository.bm25 import BM25Index, create_bm25
from simple_rag.repository.bm25.fts5_bm25 import FTS5BM25Index
from simple_rag.repository.bm25.okapi_bm25 import RankBM25Index
from simple_rag.repository.bm25.own_bm25 import OwnBM25Index
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


# ---------------------------------------------------------------- own_bm25


def test_own_bm25_路由到_OwnBM25Index(database, tok):
    index = create_bm25("own_bm25", tokenizer=tok, conn=database.connection)
    assert isinstance(index, OwnBM25Index)
    assert isinstance(index, BM25Index)


def test_own_bm25_缺_conn_报错(tok):
    with pytest.raises(ValueError, match="conn"):
        create_bm25("own_bm25", tokenizer=tok)


def test_own_bm25_传_persist_path_报错(database, tok):
    with pytest.raises(ValueError, match="persist_path"):
        create_bm25(
            "own_bm25", tokenizer=tok, conn=database.connection, persist_path="x.pkl"
        )


@pytest.mark.parametrize("k1", [0, 1e-300, 1.5, 1e308])
@pytest.mark.parametrize("b", [0, 1])
def test_own_bm25_参数边界接受并给出有限分(database, tok, k1, b):
    """k1=0 合法（与 rank_bm25 的域一致）；1e308 是**有限但极大**的上界。"""
    index = create_bm25(
        "own_bm25", tokenizer=tok, conn=database.connection, k1=k1, b=b
    )
    index.open()
    index.rebuild([{"chunk_id": "a", "text": "检索 检索"}, {"chunk_id": "b", "text": "检索"}])
    hits = index.search("检索", k=5)
    assert hits
    assert all(math.isfinite(h.score) and h.score < 0 for h in hits)


@pytest.mark.parametrize(
    "k1", [-0.1, float("nan"), float("inf"), float("-inf"), True, "1.5", 10**400]
)
def test_own_bm25_非法k1报错(database, tok, k1):
    """NaN 必须拦 —— `nan < 0` 是 False，放过去打分就会静默变 `nan`。

    `None` 不在非法之列：它是「没传」的哨兵，落到默认 1.5。
    """
    with pytest.raises(ValueError, match="k1"):
        create_bm25("own_bm25", tokenizer=tok, conn=database.connection, k1=k1)


@pytest.mark.parametrize("b", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_own_bm25_非法b报错(database, tok, b):
    with pytest.raises(ValueError, match="b"):
        create_bm25("own_bm25", tokenizer=tok, conn=database.connection, b=b)


def test_own_bm25_缺省参数等价于显式_1_5_和_0_75(database, tok):
    conn = database.connection
    default = create_bm25("own_bm25", tokenizer=tok, conn=conn)
    default.open()
    default.rebuild([{"chunk_id": "a", "text": "检索 检索 是"}, {"chunk_id": "b", "text": "检索"}])
    explicit = create_bm25("own_bm25", tokenizer=tok, conn=conn, k1=1.5, b=0.75)
    assert default.search("检索", k=5) == explicit.search("检索", k=5)


def test_own_bm25_换参数不用重建(database, tok):
    """同一份落盘数据，两个不同参数的实例各按自己的公式打分。

    参数只进打分公式，不参与落盘 —— 这是相对 `fts5` 的核心收益。
    """
    conn = database.connection
    writer = create_bm25("own_bm25", tokenizer=tok, conn=conn)
    writer.open()
    writer.rebuild(
        [
            {"chunk_id": "long", "text": "检索 是 检索 是 检索"},
            {"chunk_id": "short", "text": "检索"},
        ]
    )
    wide = create_bm25("own_bm25", tokenizer=tok, conn=conn, k1=0.3, b=0)
    narrow = create_bm25("own_bm25", tokenizer=tok, conn=conn, k1=2, b=1)
    # b=0 时不看文档长度，只按 tf 判：长文档 tf 更高 -> 更相关
    assert [h.chunk_id for h in wide.search("检索", k=5)] == ["long", "short"]
    # b=1 时长度归一化最强，短文档更相关
    assert [h.chunk_id for h in narrow.search("检索", k=5)] == ["short", "long"]
