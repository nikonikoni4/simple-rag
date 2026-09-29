"""检索层的类型。

- `RetrieverConfig` —— 启用哪个检索方式（RetrievalClient 的配置项）
- `RetrievalHit` —— 检索方式的统一中间结果，只用于融合，不对外
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class RetrieverConfig:
    """启用哪个检索方式。

    vec:  config 可给 {"dimensions": 1024|2048}，必须与建表 Schema.dim
          一致，否则查询向量维度对不上，search 直接报错。
    bm25: 无可配参数。k1/b 是 BM25Index.search 不存在的死配置，已删；
          等存储层真支持了再加。
    """

    name: Literal["bm25", "vec"]
    config: dict | None = None


@dataclass(frozen=True)
class RetrievalHit:
    """检索方式的统一中间结果 —— 只用于融合，不对外。

    score 是原始分：vec 是 distance，bm25 是负分。都越小越好，
    但单位不同，禁止跨来源比较大小。
    """

    chunk_id: str
    score: float
    retriever: str  # "vec" / "bm25"，标记这条命中来自哪一路


@dataclass(frozen=True)
class RetrievalResult:
    """search 的最终返回 —— RRF 融合后回 vec0 补过正文的命中。

    fields 是 vec0 里这一行的全部业务字段，正文在内。
    """

    chunk_id: str
    score: float  # RRF 累计分，越大越靠前
    fields: dict[str, str | None]
