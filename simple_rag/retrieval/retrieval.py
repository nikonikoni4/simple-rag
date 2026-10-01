"""综合检索。

数据表是**内容仓库**（必然需要，把 chunk_id 变回正文与来源）；
vec_db 只是**检索方式之一**（配置了 "vec" 才跑向量检索）。bm25 是
"另一种产生 chunk_id 的方式"，它的分数只参与融合，不直接暴露。

两个适配器只做翻译：把各存储的返回变成统一的 RetrievalHit。

全链路异步：embed 是网络调用，async 从 `VecRetriever` 传到 `RetrievalClient`，
各路召回用 `asyncio.gather` 并发跑。
"""

from __future__ import annotations

import asyncio

from simple_rag.embedding_api import DoubaoEmbeddingVision, TextPart
from simple_rag.repository.bm25 import BM25Index
from simple_rag.repository.chunk_store import ChunkRow, ChunkStore
from simple_rag.repository.vec import VecDB

from .types import RetrievalHit, RetrieverConfig, RetrievalResult, SourceRef

_RRF_K = 60  # RRF 标准常数


class VecRetriever:
    """稠密检索：query → embed → vec_db.search → RetrievalHit。"""

    def __init__(
        self,
        vec_db: VecDB,
        embedding_client: DoubaoEmbeddingVision,
        dimensions: int | None,
    ) -> None:
        """初始化。

        Args:
            vec_db: 向量库，检索走它。
            embedding_client: 把查询原文变成向量的 embed 客户端。
            dimensions: embed 输出维度，必须与建表 Schema.dim 一致，
                否则查询向量维度对不上、search 直接报错；
                None 表示用 embed 客户端的默认维度。
        """
        self._vec_db = vec_db
        self._embedding_client = embedding_client
        self._dimensions = dimensions

    async def search(self, query: str, k: int) -> list[RetrievalHit]:
        """检索一路。

        Args:
            query: 查询原文，先 embed 成向量。
            k: 本路召回条数。

        Returns:
            命中列表，顺序即相关顺序（distance 越小越相关），
            score 是原始 distance。
        """
        vec = (
            await self._embedding_client.embed(
                [TextPart(query)], dimensions=self._dimensions
            )
        ).dense
        hits = self._vec_db.search(vec, k)
        return [RetrievalHit(h.chunk_id, h.distance, "vec") for h in hits]


class BM25Retriever:
    """关键词检索：query 原文 → bm25_index.search → RetrievalHit。"""

    def __init__(self, bm25_index: BM25Index) -> None:
        """初始化。

        Args:
            bm25_index: BM25 全文索引。
        """
        self._bm25_index = bm25_index

    async def search(self, query: str, k: int) -> list[RetrievalHit]:
        """检索一路。

        **签名是协程只为满足 `Retriever` 协议** —— 本路走 sqlite 本地调用，
        没有可挂起的 I/O，所以函数体是同步直调，不套 `asyncio.to_thread`。

        Args:
            query: 查询原文，分词由索引内部做。
            k: 本路召回条数。

        Returns:
            命中列表，顺序即相关顺序（score 越小越相关），
            score 是原始负分。
        """
        hits = self._bm25_index.search(query, k)
        return [RetrievalHit(h.chunk_id, h.score, "bm25") for h in hits]


class RetrievalClient:
    """综合检索的编排者：粗排（各路召回）→ 精排（融合排序 + 补内容）。

    检索分两阶段，条数分开控制：
    - 粗排：每个检索器各取 coarse_top_k 条候选，只求召回不求精确排序；
    - 精排：RRF 融合候选 → 截前 k 条 → 回**数据表**补正文与来源，k 是 search 的参数。

    coarse_top_k 小于 search 的 k 时，融合池不够深，返回条数会少于 k。

    **内容仓库是数据表**（`chunk_store`，必填），不是向量表：向量表只回答
    「哪些像」，一个 chunk 是什么内容、来自哪里，都在数据表里。
    vec_db / embedding_client / bm25_index **按需注入** —— 哪条路启用了才要哪个。
    """

    def __init__(
        self,
        chunk_store: ChunkStore,
        retrievers: list[RetrieverConfig],
        coarse_top_k: int,
        *,
        vec_db: VecDB | None = None,
        embedding_client: DoubaoEmbeddingVision | None = None,
        bm25_index: BM25Index | None = None,
    ) -> None:
        """初始化并按配置建好各路检索器。

        Args:
            chunk_store: 数据表，内容仓库。无论启用哪些检索方式都要给 ——
                融合完要靠它把 chunk_id 变回正文与来源。
            retrievers: 启用哪些检索方式，至少一项。
            coarse_top_k: 粗排时每个检索器的召回条数。小于 search 的 k 时
                融合池不够深，返回条数会少于 k。
            vec_db: "vec" 路需要；启用了 vec 而没注入就在构造时报错。
            embedding_client: "vec" 路需要；启用了 vec 而没注入就在构造时报错。
            bm25_index: "bm25" 路需要；启用了 bm25 而没注入就在构造时报错。

        Raises:
            ValueError: retrievers 为空、检索器名未知、或对应依赖没注入。
        """
        self.chunk_store = chunk_store
        self.vec_db = vec_db
        self.embedding_client = embedding_client
        self.bm25_index = bm25_index
        self._retrievers: list = []
        self.coarse_top_k = coarse_top_k
        self._init_retriever(retrievers)

    def _init_retriever(self, configs: list[RetrieverConfig]) -> None:
        """按配置逐个建检索器。

        Raises:
            ValueError: 配置为空，或检索器名未知。
        """
        if not configs:
            raise ValueError("无效的检索器配置：至少要启用一个检索器")
        builders = {
            "vec": self._build_vec,
            "bm25": self._build_bm25,
        }
        for cfg in configs:
            builder = builders.get(cfg.name)
            if builder is None:
                raise ValueError(
                    f"未知的检索器名：{cfg.name!r}，可选 {sorted(builders)}"
                )
            self._retrievers.append(builder(cfg.config))

    def _build_vec(self, config: dict | None) -> VecRetriever:
        """建 vec 检索器，dimensions 从 config 取。

        Raises:
            ValueError: 启用了 vec 但没注入 embedding_client。
        """
        if self.vec_db is None:
            raise ValueError("启用了 vec 检索器，但没注入 vec_db")
        if self.embedding_client is None:
            raise ValueError("启用了 vec 检索器，但没注入 embedding_client")
        return VecRetriever(
            self.vec_db, self.embedding_client, (config or {}).get("dimensions")
        )

    def _build_bm25(self, config: dict | None) -> BM25Retriever:
        """建 bm25 检索器，无可配参数。

        Raises:
            ValueError: 启用了 bm25 但没注入 bm25_index。
        """
        if self.bm25_index is None:
            raise ValueError("启用了 bm25 检索器，但没注入 bm25_index")
        return BM25Retriever(self.bm25_index)

    # ---------------------------------------------------------------- 检索

    async def search(self, query: str, k: int) -> list[RetrievalResult]:
        """综合检索：各路粗排取 coarse_top_k 条 → RRF 融合 → 截前 k 条 → 回数据表补内容。

        各路召回**并发**跑（`asyncio.gather`）—— vec 路要等 embed 的网络往返，
        串行的话 bm25 路只能干等。

        Args:
            query: 查询原文，各路共用。
            k: 最终返回条数（精排截断）；各路召回多少由构造时的
                coarse_top_k 决定，两者分开。

        Returns:
            按 RRF 融合分降序的结果，`content` 是正文、`sources` 是来源
            （跨文件合并的块会有多条）。两表数据不一致时（召回有、数据表无）
            该 chunk 被静默丢弃，返回条数可能少于 k —— 数据一致性归写入方管，
            检索层不修。
        """
        lists = list(
            await asyncio.gather(
                *(r.search(query, self.coarse_top_k) for r in self._retrievers)
            )
        )
        fused = self._rrf(lists)
        results = []
        for chunk_id, score in fused[:k]:
            rows = self.chunk_store.get(chunk_id)
            if not rows:
                continue
            results.append(_to_result(chunk_id, score, rows))
        return results

    @staticmethod
    def _rrf(lists: list[list[RetrievalHit]]) -> list[tuple[str, float]]:
        """倒数排名融合：只看每路内部的排名，不看分值。

        Args:
            lists: 各路召回结果，列表顺序即该路的相关顺序。

        Returns:
            (chunk_id, RRF 累计分) 按分数降序。第 i 名得
            1/(_RRF_K + i + 1)，同一 chunk 多路命中则累加。
        """
        acc: dict[str, float] = {}
        for hits in lists:
            for rank, hit in enumerate(hits):
                acc[hit.chunk_id] = (
                    acc.get(hit.chunk_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
                )
        return sorted(acc.items(), key=lambda kv: -kv[1])


def _to_result(chunk_id: str, score: float, rows: list[ChunkRow]) -> RetrievalResult:
    """把数据表的一组行压成一条结果。

    正文与扩展列取**第一行** —— 同一个 chunk 的各行在这几列上是同一份
    （写入时就在逐行重复），只有 `path` 与行区间逐行不同。
    """
    first = rows[0]
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        content=first.content,
        sources=[SourceRef(row.path, row.start_line, row.end_line) for row in rows],
        fields=first.fields,
    )
