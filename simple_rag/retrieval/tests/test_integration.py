"""端到端 —— 真 Database + 真 VecDB + 真 BM25Index + 假 embed。

前三个循环用替身锁了检索层自己的逻辑；这里验证真接线：
两路检索器跑在真库上，RRF 融合后回 vec0 取回正文。
embed 是网络客户端，继续用固定向量替身。
"""

from __future__ import annotations

import asyncio
import hashlib

import pytest

from simple_rag.db import Database
from simple_rag.repository.bm25 import create_bm25
from simple_rag.repository.vec import Schema, VecDB
from simple_rag.retrieval.retrieval import RetrievalClient
from simple_rag.retrieval.types import RetrieverConfig
from simple_rag.tokenization import TokenizerFactory

from _fakes import FakeEmbedding

DIM = 4

TEXTS = [
    ("向量检索很好用", [1.0, 0.0, 0.0, 0.0]),
    ("关键词检索也不错", [0.0, 1.0, 0.0, 0.0]),
    ("今天天气如何", [0.0, 0.0, 1.0, 0.0]),
]


def cid_of(text: str) -> str:
    """32 位 hex —— fts5 实现要求 chunk_id 是 15 位以上 hex。

    不能用 "c1" 这种短 id：写库前就抛。
    """
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


@pytest.fixture
def database(tmp_path):
    database = Database(tmp_path / "retrieval.db")
    yield database
    database.close()


@pytest.fixture
def client(database):
    vec_db = VecDB(database.connection)
    vec_db.create_table(Schema(dim=DIM, metric="L2", fields=("content",)))
    vec_db.insert(
        {"chunk_id": cid_of(text), "vector": vec, "content": text}
        for text, vec in TEXTS
    )

    bm25_index = create_bm25(
        "fts5",
        tokenizer=TokenizerFactory.create("jieba"),
        conn=database.connection,
    )
    bm25_index.open()
    bm25_index.insert(
        {"chunk_id": cid_of(text), "text": text} for text, _ in TEXTS
    )

    return RetrievalClient(
        vec_db,
        [RetrieverConfig(name="vec"), RetrieverConfig(name="bm25")],
        coarse_top_k=10,
        # 查询向量 = 第一条的向量 → vec 路第一名必然是它
        embedding_client=FakeEmbedding([1.0, 0.0, 0.0, 0.0]),
        bm25_index=bm25_index,
    )


def test_端到端_两路真库融合并返回正文(client):
    results = asyncio.run(client.search("检索", k=3))

    # 三条全回来（k=3），fields 是 vec0 里的业务字段
    assert len(results) == 3
    assert {r.chunk_id for r in results} == {cid_of(text) for text, _ in TEXTS}
    for r in results:
        expected = next(text for text, _ in TEXTS if cid_of(text) == r.chunk_id)
        assert r.fields == {"content": expected}
    # 前两条两路都命中（正文都含「检索」），融合分必然高于只被 vec 命中的第三条
    assert results[0].score > results[2].score
    assert results[1].score > results[2].score
    # 查询向量与第一条完全相同 → vec 路第一 → 融合榜首
    assert results[0].chunk_id == cid_of("向量检索很好用")
