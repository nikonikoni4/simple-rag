"""综合检索 —— 以 vec 为中心。

vec_db 两个角色：内容仓库（必然需要，把 chunk_id 变回正文）；
检索方式之一（配置了 "vec" 才跑向量检索）。bm25 是"另一种产生
chunk_id 的方式"，它的分数只参与融合，不直接暴露。

两个适配器只做翻译：把各存储的返回变成统一的 RetrievalHit。
"""

from __future__ import annotations

from simple_rag.embedding_api import DoubaoEmbeddingVision, TextPart
from simple_rag.repository.bm25 import BM25Index
from simple_rag.repository.vec import VecDB

from .types import RetrievalHit, RetrieverConfig, RetrievalResult

_RRF_K = 60  # RRF 标准常数


class VecRetriever:
    """稠密检索：query → embed → vec_db.search → RetrievalHit。"""

    def __init__(
        self,
        vec_db: VecDB,
        embedding_client: DoubaoEmbeddingVision,
        dimensions: int | None,
    ) -> None:
        self._vec_db = vec_db
        self._embedding_client = embedding_client
        self._dimensions = dimensions

    def search(self, query: str, k: int) -> list[RetrievalHit]:
        vec = self._embedding_client.embed(
            [TextPart(query)], dimensions=self._dimensions
        ).dense
        hits = self._vec_db.search(vec, k)
        return [RetrievalHit(h.chunk_id, h.distance, "vec") for h in hits]


class BM25Retriever:
    """关键词检索：query 原文 → bm25_index.search → RetrievalHit。"""

    def __init__(self, bm25_index: BM25Index) -> None:
        self._bm25_index = bm25_index

    def search(self, query: str, k: int) -> list[RetrievalHit]:
        hits = self._bm25_index.search(query, k)
        return [RetrievalHit(h.chunk_id, h.score, "bm25") for h in hits]


class RetrievalClient:
    """综合检索的编排者：建检索器 → 各查一路 → RRF 融合 → 回 vec0 补正文。

    vec_db 必填（内容仓库，无论启用哪些检索方式）；
    embedding_client / bm25_index 按需注入，缺什么在构造时报什么。
    """

    def __init__(
        self,
        vec_db: VecDB,
        retrievers: list[RetrieverConfig],
        *,
        embedding_client: DoubaoEmbeddingVision | None = None,
        bm25_index: BM25Index | None = None,
    ) -> None:
        self.vec_db = vec_db
        self.embedding_client = embedding_client
        self.bm25_index = bm25_index
        self._retrievers: list = []
        self._init_retriever(retrievers)

    def _init_retriever(self, configs: list[RetrieverConfig]) -> None:
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
        if self.embedding_client is None:
            raise ValueError("启用了 vec 检索器，但没注入 embedding_client")
        return VecRetriever(
            self.vec_db, self.embedding_client, (config or {}).get("dimensions")
        )

    def _build_bm25(self, config: dict | None) -> BM25Retriever:
        if self.bm25_index is None:
            raise ValueError("启用了 bm25 检索器，但没注入 bm25_index")
        return BM25Retriever(self.bm25_index)

    # ---------------------------------------------------------------- 检索

    def search(self, query: str, k: int) -> list[RetrievalResult]:
        """综合检索：各路取 k 条候选 → RRF 融合 → 截前 k → 回 vec0 补正文。

        两表数据不一致时（bm25 有、vec0 无）该 chunk 被静默丢弃，
        返回条数可能少于 k —— 数据一致性归写入方管，检索层不修。
        """
        lists = [r.search(query, k) for r in self._retrievers]
        fused = self._rrf(lists)
        results = []
        for chunk_id, score in fused[:k]:
            row = self.vec_db.get(chunk_id)
            if row is None:
                continue
            results.append(RetrievalResult(chunk_id, score, row.fields))
        return results

    @staticmethod
    def _rrf(lists: list[list[RetrievalHit]]) -> list[tuple[str, float]]:
        """倒数排名融合：只看每路内部的排名，不看分值。

        第 i 名得 1/(_RRF_K + i + 1)，同一 chunk 多路命中则累加。
        """
        acc: dict[str, float] = {}
        for hits in lists:
            for rank, hit in enumerate(hits):
                acc[hit.chunk_id] = (
                    acc.get(hit.chunk_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
                )
        return sorted(acc.items(), key=lambda kv: -kv[1])
