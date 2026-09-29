"""RetrievalClient —— 构造时的配置分发与依赖校验，以及 search 的行为。

vec_db 是必然依赖（内容仓库）；embedding_client / bm25_index
按启用的检索方式按需注入，缺什么在构造时报什么，不拖到 search。

search 的融合期望值全部手工算好（RRF：第 i 名得 1/(60+i+1)，
多路命中累加），不从实现反推 —— 那是同义反复，测不出错。
"""

from __future__ import annotations

import pytest

from simple_rag.retrieval.retrieval import RetrievalClient
from simple_rag.retrieval.types import RetrieverConfig

from _fakes import FakeBM25Index, FakeEmbedding, FakeVecDB, bm25_row, vec_row


def test_空配置报错():
    with pytest.raises(ValueError, match="至少要启用一个"):
        RetrievalClient(FakeVecDB(), [])


def test_未知检索器名报错():
    with pytest.raises(ValueError, match="未知"):
        RetrievalClient(FakeVecDB(), [RetrieverConfig(name="fts4")])


def test_配vec但没注入embedding_client报错():
    with pytest.raises(ValueError, match="embedding_client"):
        RetrievalClient(FakeVecDB(), [RetrieverConfig(name="vec")])


def test_配bm25但没注入bm25_index报错():
    with pytest.raises(ValueError, match="bm25_index"):
        RetrievalClient(FakeVecDB(), [RetrieverConfig(name="bm25")])


def test_两路都配_依赖齐则构造成功():
    client = RetrievalClient(
        FakeVecDB(),
        [RetrieverConfig(name="vec"), RetrieverConfig(name="bm25")],
        embedding_client=FakeEmbedding([1.0]),
        bm25_index=FakeBM25Index(),
    )
    assert client is not None


# ---------------------------------------------------------------- search


def make_two_way(vec_hits, bm25_hits, rows):
    return RetrievalClient(
        FakeVecDB(vec_hits, rows),
        [RetrieverConfig(name="vec"), RetrieverConfig(name="bm25")],
        embedding_client=FakeEmbedding([1.0]),
        bm25_index=FakeBM25Index(bm25_hits),
    )


def test_融合顺序_手算期望值():
    """vec 路：a(第1) b(第2) c(第3)；bm25 路：b(第1) d(第2) a(第3)。

    第 i 名得 1/(60+i+1)，手算合计：
      b = 1/62 + 1/61；a = 1/61 + 1/63；d = 1/62；c = 1/63
    → 顺序 b > a > d > c；两路同时命中的 b、a 各出现一次。
    """
    client = make_two_way(
        [
            vec_row("a", distance=0.1),
            vec_row("b", distance=0.5),
            vec_row("c", distance=0.9),
        ],
        [bm25_row("b", -0.5), bm25_row("d", -2.0), bm25_row("a", -4.0)],
        rows={cid: vec_row(cid) for cid in "abcd"},
    )
    results = client.search("q", k=4)

    assert [r.chunk_id for r in results] == ["b", "a", "d", "c"]
    assert results[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert results[1].score == pytest.approx(1 / 61 + 1 / 63)
    assert results[2].score == pytest.approx(1 / 62)
    assert results[3].score == pytest.approx(1 / 63)


def test_补正文字段来自vec0():
    client = make_two_way(
        [vec_row("a", distance=0.1)],
        [],
        rows={"a": vec_row("a", content="真正的正文")},
    )
    results = client.search("q", k=3)
    assert results[0].fields == {"content": "真正的正文"}


def test_bm25命中但vec0缺行_静默丢弃():
    """两表数据不一致（bm25 有、vec0 无）：不抛错，缺的 chunk 不出现。"""
    client = make_two_way(
        [vec_row("a")],
        [bm25_row("x")],
        rows={"a": vec_row("a")},  # x 在 vec0 里没有
    )
    results = client.search("q", k=5)
    assert [r.chunk_id for r in results] == ["a"]


def test_融合后截断到k():
    client = make_two_way(
        [
            vec_row("a", distance=0.1),
            vec_row("b", distance=0.5),
            vec_row("c", distance=0.9),
        ],
        [bm25_row("b", -0.5), bm25_row("d", -2.0), bm25_row("a", -4.0)],
        rows={cid: vec_row(cid) for cid in "abcd"},
    )
    results = client.search("q", k=2)
    assert [r.chunk_id for r in results] == ["b", "a"]


def test_只配vec一路_不碰bm25():
    bm25 = FakeBM25Index()
    client = RetrievalClient(
        FakeVecDB(
            [vec_row("a", distance=0.1), vec_row("b", distance=0.9)],
            {"a": vec_row("a"), "b": vec_row("b")},
        ),
        [RetrieverConfig(name="vec")],
        embedding_client=FakeEmbedding([1.0]),
        bm25_index=bm25,  # 注入了但没配 → 不该被调
    )
    results = client.search("q", k=2)
    assert [r.chunk_id for r in results] == ["a", "b"]
    assert bm25.calls == []


def test_k透传到各路():
    vec_db = FakeVecDB([vec_row("a")], {"a": vec_row("a")})
    bm25 = FakeBM25Index([bm25_row("a")])
    client = RetrievalClient(
        vec_db,
        [RetrieverConfig(name="vec"), RetrieverConfig(name="bm25")],
        embedding_client=FakeEmbedding([1.0]),
        bm25_index=bm25,
    )
    client.search("q", k=6)
    assert vec_db.calls[0][1] == 6
    assert bm25.calls == [("q", 6)]
