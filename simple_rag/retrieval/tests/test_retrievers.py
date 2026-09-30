"""两个检索器适配器 —— 只测「存储结果 → RetrievalHit」的翻译。

- vec 路：query 要先过 embed 变成向量，distance 原样进 hit
- bm25 路：query 是原文，分数原样进 hit
- 两者都满足 base.Retriever 契约
"""

from __future__ import annotations

import asyncio

from simple_rag.embedding_api import TextPart
from simple_rag.retrieval.base import Retriever
from simple_rag.retrieval.retrieval import BM25Retriever, VecRetriever
from simple_rag.retrieval.types import RetrievalHit

from _fakes import FakeBM25Index, FakeEmbedding, FakeVecDB, bm25_row, vec_row


# ---------------------------------------------------------------- vec 适配器


def test_vec_翻译距离为hit且参数透传():
    fake_db = FakeVecDB([vec_row("a", distance=0.1), vec_row("b", distance=0.9)])
    embedding = FakeEmbedding([0.1, 0.2])
    retriever = VecRetriever(fake_db, embedding, dimensions=8)

    hits = asyncio.run(retriever.search("向量检索", k=5))

    assert hits == [
        RetrievalHit("a", 0.1, "vec"),
        RetrievalHit("b", 0.9, "vec"),
    ]
    # embed 收到 TextPart(query)，dimensions 原样透传
    parts, dims = embedding.calls[0]
    assert isinstance(parts[0], TextPart)
    assert parts[0].text == "向量检索"
    assert dims == 8
    # 查询向量原样进了 vec_db.search，k 原样透传
    assert fake_db.calls == [([0.1, 0.2], 5)]
    assert isinstance(retriever, Retriever)


def test_vec_不配dimensions则embed收None():
    embedding = FakeEmbedding([0.0])
    asyncio.run(VecRetriever(FakeVecDB([]), embedding, None).search("q", 3))
    assert embedding.calls[0][1] is None


# ---------------------------------------------------------------- bm25 适配器


def test_bm25_翻译分数为hit且参数透传():
    fake = FakeBM25Index([bm25_row("a", -1.2), bm25_row("b", -3.4)])

    hits = asyncio.run(BM25Retriever(fake).search("关键词", 7))

    assert hits == [
        RetrievalHit("a", -1.2, "bm25"),
        RetrievalHit("b", -3.4, "bm25"),
    ]
    assert fake.calls == [("关键词", 7)]
