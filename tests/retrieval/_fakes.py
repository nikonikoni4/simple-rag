"""测试替身 —— 只实现被测代码用到的接口。

真存储（VecDB / ChunkStore / BM25Index）在 repository 各自的 tests 里有测试；
embed 客户端是网络客户端，这里用固定向量替身。
"""

from __future__ import annotations

from types import SimpleNamespace

from simple_rag.repository.bm25 import BM25SearchResult
from simple_rag.repository.chunk_store import ChunkRow
from simple_rag.repository.vec import VecSearchResult


def vec_row(cid, *, distance=0.5, fields=None, content=None):
    """构造一行 `VecSearchResult`。`content` 是生成 fields 的便捷参数。"""
    if fields is None:
        fields = {"content": content if content is not None else f"正文-{cid}"}
    return VecSearchResult(
        chunk_id=cid,
        distance=distance,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        fields=fields,
    )


def bm25_row(cid, score=-1.0):
    return BM25SearchResult(chunk_id=cid, score=score)


def chunk_row(
    cid, *, content=None, path="doc.md", start_line=0, end_line=9, fields=None
):
    """构造一行 `ChunkRow` —— 数据表按 chunk_id 取回的一行（一条来源）。"""
    return ChunkRow(
        chunk_id=cid,
        path=path,
        start_line=start_line,
        end_line=end_line,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        content=content if content is not None else f"正文-{cid}",
        fields=fields if fields is not None else {},
    )


class FakeEmbedding:
    """`embed` 永远返回同一个向量，记录每次调用。协程 —— 与真客户端签名一致。"""

    def __init__(self, dense):
        self._dense = dense
        self.calls: list = []

    async def embed(self, parts, *, dimensions=None, **_):
        self.calls.append((parts, dimensions))
        return SimpleNamespace(dense=self._dense)


class FakeVecDB:
    """`search` 返回预设的行；`get` 按字典查，缺的给 `None`（模拟两表不一致）。"""

    def __init__(self, search_hits=None, rows=None):
        self._search_hits = search_hits or []
        self._rows = rows or {}
        self.calls: list = []

    def search(self, vector, k):
        self.calls.append((vector, k))
        return self._search_hits[:k]

    def get(self, chunk_id):
        return self._rows.get(chunk_id)


class FakeChunkStore:
    """数据表替身：`get` 按字典查，缺的给**空列表**（模拟数据表里没有这个 chunk）。"""

    def __init__(self, rows=None):
        self._rows = rows or {}
        self.calls: list = []

    def get(self, chunk_id):
        self.calls.append(chunk_id)
        return self._rows.get(chunk_id, [])


class FakeBM25Index:
    """`search` 返回预设的命中。"""

    def __init__(self, hits=None):
        self._hits = hits or []
        self.calls: list = []

    def search(self, query, k):
        self.calls.append((query, k))
        return self._hits[:k]
