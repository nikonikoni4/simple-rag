"""RetrievalClient —— 构造时的配置分发与依赖校验，以及 search 的行为。

`chunk_store`（数据表）是必然依赖（内容仓库）；`vec_db` / `embedding_client` /
`bm25_index` 按启用的检索方式**按需注入**，缺什么在构造时报什么，不拖到 search。

search 的融合期望值全部手工算好（RRF：第 i 名得 1/(60+i+1)，
多路命中累加），不从实现反推 —— 那是同义反复，测不出错。
"""

from __future__ import annotations

import asyncio

import pytest

from simple_rag.retrieval.retrieval import RetrievalClient
from simple_rag.retrieval.types import RetrieverConfig

from _fakes import (
    FakeBM25Index,
    FakeChunkStore,
    FakeEmbedding,
    FakeVecDB,
    bm25_row,
    chunk_row,
    vec_row,
)


def _rows(*cids) -> dict:
    """把若干个 chunk_id 做成数据表的返回形状 —— 每个一块、一条来源。"""
    return {cid: [chunk_row(cid)] for cid in cids}


def test_空配置报错():
    with pytest.raises(ValueError, match="至少要启用一个"):
        RetrievalClient(FakeChunkStore(), [], coarse_top_k=10)


def test_未知检索器名报错():
    with pytest.raises(ValueError, match="未知"):
        RetrievalClient(
            FakeChunkStore(), [RetrieverConfig(name="fts4")], coarse_top_k=10
        )


def test_配vec但没注入vec_db报错():
    with pytest.raises(ValueError, match="vec_db"):
        RetrievalClient(
            FakeChunkStore(),
            [RetrieverConfig(name="vec")],
            coarse_top_k=10,
            embedding_client=FakeEmbedding([1.0]),
        )


def test_配vec但没注入embedding_client报错():
    with pytest.raises(ValueError, match="embedding_client"):
        RetrievalClient(
            FakeChunkStore(),
            [RetrieverConfig(name="vec")],
            coarse_top_k=10,
            vec_db=FakeVecDB(),
        )


def test_配bm25但没注入bm25_index报错():
    with pytest.raises(ValueError, match="bm25_index"):
        RetrievalClient(
            FakeChunkStore(), [RetrieverConfig(name="bm25")], coarse_top_k=10
        )


def test_两路都配_依赖齐则构造成功():
    client = RetrievalClient(
        FakeChunkStore(),
        [RetrieverConfig(name="vec"), RetrieverConfig(name="bm25")],
        coarse_top_k=10,
        vec_db=FakeVecDB(),
        embedding_client=FakeEmbedding([1.0]),
        bm25_index=FakeBM25Index(),
    )
    assert client is not None


# ---------------------------------------------------------------- search


def make_two_way(vec_hits, bm25_hits, rows, coarse_top_k=10):
    return RetrievalClient(
        FakeChunkStore(rows),
        [RetrieverConfig(name="vec"), RetrieverConfig(name="bm25")],
        coarse_top_k=coarse_top_k,
        vec_db=FakeVecDB(vec_hits),
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
        rows=_rows("a", "b", "c", "d"),
    )
    results = asyncio.run(client.search("q", k=4))

    assert [r.chunk_id for r in results] == ["b", "a", "d", "c"]
    assert results[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert results[1].score == pytest.approx(1 / 61 + 1 / 63)
    assert results[2].score == pytest.approx(1 / 62)
    assert results[3].score == pytest.approx(1 / 63)


def test_内容与来源来自数据表():
    client = make_two_way(
        [vec_row("a", distance=0.1)],
        [],
        rows={
            "a": [
                chunk_row(
                    "a", content="真正的正文", path="notes/a.md", start_line=3, end_line=7
                )
            ]
        },
    )
    results = asyncio.run(client.search("q", k=3))

    assert results[0].content == "真正的正文"
    assert [(s.path, s.start_line, s.end_line) for s in results[0].sources] == [
        ("notes/a.md", 3, 7)
    ]


def test_跨文件合并的块返回多条来源():
    """一个 chunk 可能来自多个文件 —— 来源是列表，不是单个路径。"""
    client = make_two_way(
        [vec_row("a", distance=0.1)],
        [],
        rows={
            "a": [
                chunk_row("a", path="a.md", start_line=1, end_line=10),
                chunk_row("a", path="b.md", start_line=5, end_line=8),
            ]
        },
    )
    results = asyncio.run(client.search("q", k=3))

    assert [s.path for s in results[0].sources] == ["a.md", "b.md"]
    assert results[0].content == "正文-a"  # 正文取第一行，各行是同一份


def test_召回有但数据表缺行_静默丢弃():
    """两表数据不一致（召回有、数据表无）：不抛错，缺的 chunk 不出现。"""
    client = make_two_way(
        [vec_row("a")],
        [bm25_row("x")],
        rows=_rows("a"),  # x 在数据表里没有
    )
    results = asyncio.run(client.search("q", k=5))
    assert [r.chunk_id for r in results] == ["a"]


def test_融合后截断到k():
    client = make_two_way(
        [
            vec_row("a", distance=0.1),
            vec_row("b", distance=0.5),
            vec_row("c", distance=0.9),
        ],
        [bm25_row("b", -0.5), bm25_row("d", -2.0), bm25_row("a", -4.0)],
        rows=_rows("a", "b", "c", "d"),
    )
    results = asyncio.run(client.search("q", k=2))
    assert [r.chunk_id for r in results] == ["b", "a"]


def test_只配vec一路_不碰bm25():
    bm25 = FakeBM25Index()
    client = RetrievalClient(
        FakeChunkStore(_rows("a", "b")),
        [RetrieverConfig(name="vec")],
        coarse_top_k=10,
        vec_db=FakeVecDB([vec_row("a", distance=0.1), vec_row("b", distance=0.9)]),
        embedding_client=FakeEmbedding([1.0]),
        bm25_index=bm25,  # 注入了但没配 → 不该被调
    )
    results = asyncio.run(client.search("q", k=2))
    assert [r.chunk_id for r in results] == ["a", "b"]
    assert bm25.calls == []


def test_粗排条数透传到各路_k只管最终截断():
    """粗排与精排的条数分开：各路拿 coarse_top_k，最终返回截到 k。"""
    vec_db = FakeVecDB([vec_row("a", distance=0.1), vec_row("b", distance=0.5)])
    bm25 = FakeBM25Index([bm25_row("a"), bm25_row("b")])
    client = RetrievalClient(
        FakeChunkStore(_rows("a", "b")),
        [RetrieverConfig(name="vec"), RetrieverConfig(name="bm25")],
        coarse_top_k=6,
        vec_db=vec_db,
        embedding_client=FakeEmbedding([1.0]),
        bm25_index=bm25,
    )
    results = asyncio.run(client.search("q", k=1))
    assert vec_db.calls[0][1] == 6
    assert bm25.calls == [("q", 6)]
    assert [r.chunk_id for r in results] == ["a"]  # a 两路都第一，融合分最高


def test_coarse_top_k小于k_返回条数受限于候选池():
    """coarse_top_k 是融合池的上限：池子比 k 浅，最终凑不满 k 条。"""
    client = make_two_way(
        [vec_row("a", distance=0.1), vec_row("b", distance=0.5)],
        [bm25_row("a")],
        rows=_rows("a", "b"),
        coarse_top_k=1,
    )
    results = asyncio.run(client.search("q", k=5))
    # 两路各只回第 1 名（都是 a）→ 池里只有 a，b 进不了池
    assert [r.chunk_id for r in results] == ["a"]


def test_各路召回并发跑而不是串行():
    """契约：各路是 `asyncio.gather` 并发 —— vec 路等网络时 bm25 路不该干等。

    验证方式：换两个会在中途让出控制权的替身，看事件顺序。
    并发 → a 和 b 的 start 都排在 end 之前；串行 → a-start,a-end,b-start,b-end。

    如果失败，说明有人把 gather 改回了串行循环 —— 那 vec 路的网络往返
    会把其余各路全部堵住。
    """
    order: list[str] = []

    class YieldingRetriever:
        """`search` 中途 `await sleep(0)` 让出控制权，好观察交错。"""

        def __init__(self, name: str) -> None:
            self._name = name

        async def search(self, query, k):
            order.append(f"{self._name}-start")
            await asyncio.sleep(0)
            order.append(f"{self._name}-end")
            return []

    client = RetrievalClient(
        FakeChunkStore(),
        [RetrieverConfig(name="vec"), RetrieverConfig(name="bm25")],
        coarse_top_k=5,
        vec_db=FakeVecDB(),
        embedding_client=FakeEmbedding([1.0]),
        bm25_index=FakeBM25Index(),
    )
    # 白盒：检索器由构造时的配置建好，这里直接换成会交错的替身
    client._retrievers = [YieldingRetriever("a"), YieldingRetriever("b")]

    asyncio.run(client.search("q", k=1))

    assert order == ["a-start", "b-start", "a-end", "b-end"]
