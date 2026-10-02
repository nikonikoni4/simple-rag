"""`RagIndexingPipeline` 的单测：三张表的事务写入、扩展列分派、按文件作废。

不联网 —— embedding 用一个只满足 `embed(...).dense` 契约的假客户端。
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from simple_rag.chunking.structured_file.types import Chunk, ChunkDraft, Segment
from simple_rag.db import Database
from simple_rag.indexing import RagIndexingPipeline, RagIndexStrategy
from simple_rag.repository.bm25 import create_bm25
from simple_rag.repository.chunk_store import ChunkStore
from simple_rag.repository.chunk_store import Schema as ChunkSchema
from simple_rag.repository.vec import Schema as VecSchema
from simple_rag.repository.vec import VecDB
from simple_rag.tokenization import Tokenizer, TokenizerFactory

DIM = 4


# ------------------------------------------------------------------ 假件


class _FakeResult:
    def __init__(self, dense: list[float]) -> None:
        self.dense = dense


class _FakeEmbedder:
    """只满足 `await embed(parts, dimensions=...) -> .dense` 这个契约。"""

    def __init__(self) -> None:
        self.texts: list[str] = []

    async def embed(self, parts, dimensions=None) -> _FakeResult:
        self.texts.extend(part.text for part in parts)
        return _FakeResult([0.1] * DIM)


# ------------------------------------------------------------------ 搭建


@dataclass
class _Setup:
    index: RagIndexingPipeline
    db: Database
    vec_schema: VecSchema
    chunk_schema: ChunkSchema
    embedder: _FakeEmbedder
    tokenizer: Tokenizer

    def vec(self) -> VecDB:
        store = VecDB(self.db.connection)
        store.create_table(self.vec_schema)
        return store

    def data(self) -> ChunkStore:
        store = ChunkStore(self.db.connection)
        store.create_table(self.chunk_schema)
        return store


def _make(
    *,
    use_vec: bool = True,
    use_bm25: bool = True,
    vec_fields: tuple[str, ...] = (),
    chunk_fields: tuple[str, ...] = (),
    min_tokens: int = 10,
    max_token: int = 512,
) -> _Setup:
    vec_schema = VecSchema(dim=DIM, metric="cosine", fields=vec_fields)
    chunk_schema = ChunkSchema(fields=chunk_fields)
    strategy = RagIndexStrategy(
        use_vec=use_vec,
        use_bm25=use_bm25,
        chunk_schema=chunk_schema,
        min_tokens=min_tokens,
        max_token=max_token,
        vec_schema=vec_schema if use_vec else None,
        bm25_policy="fts5" if use_bm25 else "",
    )
    db = Database(":memory:")
    embedder = _FakeEmbedder()
    tokenizer = TokenizerFactory.create("jieba")
    pipeline = RagIndexingPipeline(strategy, db, embedder, tokenizer)
    return _Setup(pipeline, db, vec_schema, chunk_schema, embedder, tokenizer)


def _seg(text: str, path: str = "a.md", start: int = 1, end: int = 2) -> Segment:
    return Segment(
        pref="", text=text, file_path=path, start_line=start, end_line=end, tokens=4
    )


def _chunk(text: str, path: str = "a.md", vec=None) -> Chunk:
    return Chunk([_seg(text, path)], 4, np.ones(DIM) if vec is None else vec)


# ------------------------------------------------------------------ store


def test_store把一批chunk写进三张表():
    setup = _make()
    chunk = _chunk("苹果手机很好")

    setup.index.store([chunk])

    assert setup.vec().get(chunk.chunk_id) is not None
    assert setup.data().get(chunk.chunk_id)[0].content == "苹果手机很好"
    assert [hit.chunk_id for hit in _bm25(setup).search("苹果", 5)] == [chunk.chunk_id]


def test_store的正文不带摘要():
    """数据表的 content 是纯原文；embedding 才带摘要。"""
    setup = _make()
    chunk = _chunk("正文。")

    setup.index.store([chunk])

    assert setup.data().get(chunk.chunk_id)[0].content == "正文。"


def test_store的sources按文件聚合行区间():
    """同一个文件的多个段合成一行，取 min/max 行号。"""
    setup = _make()
    chunk = Chunk(
        [_seg("甲。", "a.md", 1, 3), _seg("乙。", "b.md", 5, 6), _seg("丙。", "a.md", 9, 12)],
        4,
        np.ones(DIM),
    )

    setup.index.store([chunk])

    rows = setup.data().get(chunk.chunk_id)
    assert [(r.path, r.start_line, r.end_line) for r in rows] == [
        ("a.md", 1, 12),
        ("b.md", 5, 6),
    ]


def test_store的扩展列按各自Schema分派():
    setup = _make(vec_fields=("tag",), chunk_fields=("label",))
    chunk = _chunk("正文。")

    setup.index.store([chunk], fields_of=lambda c: {"tag": "T", "label": "L"})

    assert setup.vec().get(chunk.chunk_id).fields == {"tag": "T"}
    assert setup.data().get(chunk.chunk_id)[0].fields == {"label": "L"}


def test_没有fields_of时扩展列全部为None():
    setup = _make(vec_fields=("tag",), chunk_fields=("label",))
    chunk = _chunk("正文。")

    setup.index.store([chunk])

    assert setup.vec().get(chunk.chunk_id).fields == {"tag": None}
    assert setup.data().get(chunk.chunk_id)[0].fields == {"label": None}


def test_扩展列缺失会报错():
    setup = _make(vec_fields=("tag",))
    chunk = _chunk("正文。")

    with pytest.raises(KeyError):
        setup.index.store([chunk], fields_of=lambda c: {})


def test_同一批里重复的chunk_id只写一条():
    """`chunk_id` 按内容寻址，同内容同路径就是同一条；vec0 撞主键会抛异常。"""
    setup = _make()
    a = _chunk("正文。")
    b = _chunk("正文。")

    setup.index.store([a, b])

    assert len(setup.data().get(a.chunk_id)) == 1


def test_一批里有一条非法就整批不落库():
    setup = _make(vec_fields=("tag",))
    good = _chunk("好。")
    bad = _chunk("坏。")

    with pytest.raises(KeyError):
        setup.index.store([good, bad], fields_of=lambda c: {"tag": "T"} if c is good else {})

    assert setup.vec().get(good.chunk_id) is None
    assert setup.data().get(good.chunk_id) == []


# ------------------------------------------------------------------ delete_by_path


def test_delete_by_path把沾到的chunk整个作废():
    """跨文件合并的块沾了被改的文件时，它的全部行一起消失 —— 不留半行。"""
    setup = _make()
    merged = Chunk([_seg("甲。", "a.md", 1, 3), _seg("乙。", "b.md", 5, 6)], 4, np.ones(DIM))
    other = _chunk("丙。", "c.md")
    setup.index.store([merged, other])

    setup.index.delete_by_path("a.md")

    assert setup.vec().get(merged.chunk_id) is None
    assert setup.data().get(merged.chunk_id) == []
    assert [hit.chunk_id for hit in _bm25(setup).search("丙", 5)] == [other.chunk_id]


def test_delete_by_path不动没沾该文件的chunk():
    setup = _make()
    kept = _chunk("保留。", "b.md")
    setup.index.store([kept])

    setup.index.delete_by_path("a.md")

    assert setup.vec().get(kept.chunk_id) is not None
    assert setup.data().get(kept.chunk_id) != []


def test_delete_by_path没有命中时什么都不做():
    setup = _make()

    setup.index.delete_by_path("nope.md")  # 不该抛


# ------------------------------------------------------------------ 通道开关


def test_只开vec通道时不写bm25():
    setup = _make(use_vec=True, use_bm25=False)
    chunk = _chunk("正文。")

    setup.index.store([chunk])

    assert setup.vec().get(chunk.chunk_id) is not None
    assert setup.data().get(chunk.chunk_id) != []


def test_只开bm25通道时不写vec():
    setup = _make(use_vec=False, use_bm25=True)
    chunk = _chunk("正文。")

    setup.index.store([chunk])

    assert setup.data().get(chunk.chunk_id) != []
    assert [hit.chunk_id for hit in _bm25(setup).search("正文", 5)] == [chunk.chunk_id]


# ------------------------------------------------------------------ 按 path 过滤


def test_search_in_path只返回该文件的chunk():
    setup = _make()
    a = _chunk("苹果手机很好。", "a.md")
    b = _chunk("苹果手机很好。", "b.md")
    setup.index.store([a, b])

    hits = setup.index.search_in_path(np.ones(DIM), k=5, path="a.md")

    assert [chunk_id for chunk_id, _ in hits] == [a.chunk_id]


def test_search_in_path哪怕排不进k也要召回():
    """先按 path 筛、再在候选里找最近的。

    目标 chunk 与查询正交（最不像），k=1 时全局第一名在另一个文件 ——
    按 path 筛完只剩它，仍要返回。这是「先筛后查」与「先取 k 条再筛」的分界。
    """
    setup = _make()
    near = _chunk("苹果手机很好。", "b.md", vec=[1.0, 0.0, 0.0, 0.0])
    far = _chunk("苹果手机很好。", "a.md", vec=[0.0, 1.0, 0.0, 0.0])
    setup.index.store([near, far])

    hits = setup.index.search_in_path([1.0, 0.0, 0.0, 0.0], k=1, path="a.md")

    assert [chunk_id for chunk_id, _ in hits] == [far.chunk_id]


def test_search_in_path返回按距离升序():
    setup = _make()
    close = _chunk("甲。", "a.md", vec=[1.0, 0.1, 0.0, 0.0])
    farther = _chunk("乙。", "a.md", vec=[0.0, 0.0, 1.0, 0.0])
    setup.index.store([farther, close])

    hits = setup.index.search_in_path([1.0, 0.0, 0.0, 0.0], k=5, path="a.md")

    assert [chunk_id for chunk_id, _ in hits] == [close.chunk_id, farther.chunk_id]
    assert hits[0][1] < hits[1][1]


def test_search_in_path没启用向量通道时报错():
    setup = _make(use_vec=False, use_bm25=True)

    with pytest.raises(ValueError, match="没启用向量通道"):
        setup.index.search_in_path(np.ones(DIM), k=1, path="a.md")


# ------------------------------------------------------------------ 切分 / 合并 / 向量化


def test_chunking按文件切并记相对路径(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "b.md").write_text("# 乙\n\n正文乙。", encoding="utf-8")
    (tmp_path / "docs" / "a.md").write_text("# 甲\n\n正文甲。", encoding="utf-8")
    setup = _make()

    drafts = setup.index.chunking([Path("docs")])

    assert [d.file_path for d in drafts] == [["docs/a.md"], ["docs/b.md"]]


def test_merge把小草稿并进相邻的():
    setup = _make(min_tokens=10)
    big = ChunkDraft([_seg("甲" * 200)], 200)
    small = ChunkDraft([_seg("乙")], 1)

    assert len(setup.index.merge([big, small])) == 1


def test_embedding把草稿升成Chunk():
    setup = _make()
    drafts = [ChunkDraft([_seg("正文。")], 4)]

    chunks = asyncio.run(setup.index.embedding(drafts))

    assert len(chunks) == 1
    assert chunks[0].embedding_vec.shape == (DIM,)


def test_embedding给模型的文本带面包屑():
    """算向量用的文本带面包屑；与入库的正文刻意不同源。"""
    setup = _make()
    seg = Segment(
        pref="标题",
        text="正文。",
        file_path="a.md",
        start_line=1,
        end_line=2,
        tokens=4,
    )

    asyncio.run(setup.index.embedding([ChunkDraft([seg], 4)]))

    assert setup.embedder.texts == ["标题\n正文。"]


def test_pipeline串起四步(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # 内容要够长：太短又没邻居的块会被 merge 整个丢掉
    (tmp_path / "a.md").write_text(
        "# 甲\n\n" + "苹果手机很好。" * 50, encoding="utf-8"
    )
    setup = _make()

    asyncio.run(setup.index.pipeline([Path(".")]))

    count = setup.db.connection.execute(
        "select count(*) from chunk_data"
    ).fetchone()[0]
    assert count == 1


# ------------------------------------------------------------------ 辅助


def _bm25(setup: _Setup):
    """自己建一个 fts5 实例查同一个表 —— 不碰 pipeline 的私有属性。"""
    index = create_bm25("fts5", tokenizer=setup.tokenizer, conn=setup.db.connection)
    index.open()
    return index
