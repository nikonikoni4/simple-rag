"""三个实现的统一契约 —— 参数化跑同一套断言。

只断言**接口语义必须一致**的部分（能查回、负分、升序、空查询空结果、
空批清空、不支持的操作抛错）。不断言各实现分数或排序逐位相同 —— 三者的
idf 处理与默认 k1/b 本来就不同，那不是契约的一部分。
"""

from __future__ import annotations

import hashlib

import pytest

from simple_rag.db import Database
from simple_rag.repository.bm25 import create_bm25
from simple_rag.tokenization import TokenizerFactory

DOC = "向量检索是基于语义的检索方法"
OTHER = "BM25 是基于词频的检索方法"

IMPLS = ["fts5", "rank_bm25", "own_bm25"]


def cid(n: int) -> str:
    """32 位 hex（blake2b）—— 满足三边的要求（fts5 要 >=15 位 hex，
    rank_bm25 / own_bm25 任意非空 str 都行）。

    ⚠️ 不能用 `f"{n:032x}"`：前 15 位全 0，fts5 的 rowid 会撞成同一个。
    """
    return hashlib.blake2b(str(n).encode("utf-8"), digest_size=16).hexdigest()


@pytest.fixture
def tok():
    return TokenizerFactory.create("jieba")


@pytest.fixture(params=IMPLS)
def index(request, tok, tmp_path):
    """参数化的 `BM25Index`。fts5 / own_bm25 挂真临时库；rank_bm25 纯内存。"""
    if request.param in ("fts5", "own_bm25"):
        db = Database(tmp_path / "bm25.db")
        instance = create_bm25(request.param, tokenizer=tok, conn=db.connection)
    else:
        db = None
        instance = create_bm25("rank_bm25", tokenizer=tok)
    instance.open()
    yield instance
    if db is not None:
        db.close()


# ---------------------------------------------------------------- 查询契约


def test_rebuild后能查回(index):
    index.rebuild([{"chunk_id": cid(1), "text": DOC}])
    hits = index.search("检索", k=5)
    assert [h.chunk_id for h in hits] == [cid(1)]


def test_结果按分数升序(index):
    """「越小越相关」是方向契约 —— 符号两边不保证一致（见 BM25SearchResult）。"""
    index.rebuild([
        {"chunk_id": cid(1), "text": DOC},
        {"chunk_id": cid(2), "text": OTHER},
    ])
    hits = index.search("检索", k=5)
    assert hits
    assert [h.score for h in hits] == sorted(h.score for h in hits)


def test_只返回命中文档(index):
    """查询词一个都不在文中的文档不进结果 —— 即使 k 给得很大。"""
    index.rebuild([
        {"chunk_id": cid(1), "text": DOC},
        {"chunk_id": cid(2), "text": "完全无关的内容"},
    ])
    assert [h.chunk_id for h in index.search("检索", k=10)] == [cid(1)]


def test_不相关的查询返回空(index):
    index.rebuild([{"chunk_id": cid(1), "text": DOC}])
    assert index.search("量子力学", k=5) == []


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_空查询返回空(index, blank):
    index.rebuild([{"chunk_id": cid(1), "text": DOC}])
    assert index.search(blank, k=5) == []


def test_k_为0返回空(index):
    index.rebuild([{"chunk_id": cid(1), "text": DOC}])
    assert index.search("检索", k=0) == []


# ---------------------------------------------------------------- 写入契约


def test_rebuild_丢弃旧数据(index):
    """「语义」只在 DOC 里 —— OTHER 里也有「检索」，不能拿它当旧数据专属词。"""
    index.rebuild([{"chunk_id": cid(1), "text": DOC}])
    index.rebuild([{"chunk_id": cid(2), "text": OTHER}])
    assert index.search("语义", k=5) == []
    assert [h.chunk_id for h in index.search("词频", k=5)] == [cid(2)]


def test_rebuild_空批清空(index):
    index.rebuild([{"chunk_id": cid(1), "text": DOC}])
    assert index.rebuild([]) == 0
    assert index.search("检索", k=5) == []
