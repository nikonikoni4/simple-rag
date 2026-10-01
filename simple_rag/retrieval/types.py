"""检索层的类型。

- `RetrieverConfig` —— 启用哪个检索方式（RetrievalClient 的配置项）
- `RetrievalHit` —— 检索方式的统一中间结果，只用于融合，不对外
- `SourceRef` / `RetrievalResult` —— `search` 的最终返回
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class RetrieverConfig:
    """启用哪个检索方式（RetrievalClient 的配置项）。

    Attributes:
        name: 检索方式名，"bm25" 或 "vec"。
        config: 该路的可选参数。vec 可给 {"dimensions": 1024|2048}，
            必须与建表 Schema.dim 一致，否则查询向量维度对不上，
            search 直接报错；bm25 本身无可配参数 —— k1/b 在
            `create_bm25` 构造时按实现定：rank_bm25 可调，
            fts5 硬编码在 1.2 / 0.75、传了报错。
    """

    name: Literal["bm25", "vec"]
    config: dict | None = None


@dataclass(frozen=True)
class RetrievalHit:
    """检索方式的统一中间结果 —— 只用于融合，不对外。

    Attributes:
        chunk_id: 命中的 chunk id。
        score: 原始分 —— vec 是 distance，bm25 是负分。都越小越好，
            但单位不同，禁止跨来源比较大小。
        retriever: 这条命中来自哪一路，"vec" / "bm25"。
    """

    chunk_id: str
    score: float
    retriever: str  # "vec" / "bm25"，标记这条命中来自哪一路


@dataclass(frozen=True)
class SourceRef:
    """一条来源：哪个文件的哪几行。

    跨文件合并会让一个 chunk 有多条来源，所以 `RetrievalResult.sources` 是列表。
    """

    path: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class RetrievalResult:
    """search 的最终返回 —— RRF 融合后回**数据表**补齐正文与来源的命中。

    Attributes:
        chunk_id: 命中的 chunk id。
        score: RRF 累计分，越大越靠前。
        content: 正文。
        sources: 来源列表，按 `path` / `start_line` 排好序。跨文件合并的块会有多条。
        fields: 调用方在数据表上声明的扩展列。
    """

    chunk_id: str
    score: float  # RRF 累计分，越大越靠前
    content: str
    sources: list[SourceRef]
    fields: dict[str, str | None]
