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

    def search(self, query: str, k: int) -> list[RetrievalHit]:
        """检索一路。

        Args:
            query: 查询原文，先 embed 成向量。
            k: 本路召回条数。

        Returns:
            命中列表，顺序即相关顺序（distance 越小越相关），
            score 是原始 distance。
        """
        vec = self._embedding_client.embed(
            [TextPart(query)], dimensions=self._dimensions
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

    def search(self, query: str, k: int) -> list[RetrievalHit]:
        """检索一路。

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
    """综合检索的编排者：粗排（各路召回）→ 精排（融合排序 + 补正文）。

    检索分两阶段，条数分开控制：
    - 粗排：每个检索器各取 coarse_top_k 条候选，只求召回不求精确排序；
    - 精排：RRF 融合候选 → 截前 k 条 → 回 vec0 补正文，k 是 search 的参数。

    coarse_top_k 小于 search 的 k 时，融合池不够深，返回条数会少于 k。

    vec_db 必填（内容仓库，无论启用哪些检索方式）；
    embedding_client / bm25_index 按需注入，缺什么在构造时报什么。
    """

    def __init__(
        self,
        vec_db: VecDB,
        retrievers: list[RetrieverConfig],
        coarse_top_k: int,
        *,
        embedding_client: DoubaoEmbeddingVision | None = None,
        bm25_index: BM25Index | None = None,
    ) -> None:
        """初始化并按配置建好各路检索器。

        Args:
            vec_db: 内容仓库兼向量检索的存储，无论启用哪些检索方式都要给。
            retrievers: 启用哪些检索方式，至少一项。
            coarse_top_k: 粗排时每个检索器的召回条数。小于 search 的 k 时
                融合池不够深，返回条数会少于 k。
            embedding_client: "vec" 路需要；启用了 vec 而没注入就在构造时报错。
            bm25_index: "bm25" 路需要；启用了 bm25 而没注入就在构造时报错。

        Raises:
            ValueError: retrievers 为空、检索器名未知、或对应依赖没注入。
        """
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

    def search(self, query: str, k: int) -> list[RetrievalResult]:
        """综合检索：各路粗排取 coarse_top_k 条 → RRF 融合 → 截前 k 条 → 回 vec0 补正文。

        Args:
            query: 查询原文，各路共用。
            k: 最终返回条数（精排截断）；各路召回多少由构造时的
                coarse_top_k 决定，两者分开。

        Returns:
            按 RRF 融合分降序的结果，fields 是 vec0 里该行的全部业务字段
            （正文在内）。两表数据不一致时（bm25 有、vec0 无）该 chunk
            被静默丢弃，返回条数可能少于 k —— 数据一致性归写入方管，
            检索层不修。
        """
        lists = [r.search(query, self.coarse_top_k) for r in self._retrievers]
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
