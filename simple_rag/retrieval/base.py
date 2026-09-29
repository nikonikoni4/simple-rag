"""检索器的统一契约 —— 检索方式的公共接口。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import RetrievalHit


@runtime_checkable
class Retriever(Protocol):
    """一种检索方式：把 query 变成一组带原始分的命中。

    score 的语义由实现自己定义（vec 是 distance，bm25 是负分），
    只保证「在单路结果内越小越相关」；跨来源比较没有意义，
    融合用的是排名，不是分值。
    """

    def search(self, query: str, k: int) -> list[RetrievalHit]:
        """检索一路。

        Args:
            query: 查询原文。
            k: 本路召回条数。

        Returns:
            命中列表，列表顺序即该路的相关顺序（score 越小越相关）。
        """
