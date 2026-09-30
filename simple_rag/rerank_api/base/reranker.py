"""`Reranker` —— 统一接口的抽象基类。

调用方只看得到本类，不感知具体厂商。实现各自内聚在自己的模块里
（如 `aliyun.py` 的 `AliyunReranker`）。

**接口语义（所有实现必须一致的部分）：**

- `query` 是**原文**，一切预处理（截断、清洗）在实现内部做。
- `documents` 为空 → 返回 `[]`，**不得发请求**。
- 返回列表**顺序即相关度顺序（最相关在前），不含分数** —— 分数方向（各家
  API 越大 / 越小越相关）是厂商差异，统一方向是实现的职责，调用方只拿排序。
- `RerankDocument.chunk_id` 原样透传到 `RerankHit.chunk_id`，不去重、不改写。
- `top_n` 限定返回条数；`None` = 全量返回。
- `aclose()` 幂等：无持久资源的实现可以是 no-op，但必须存在 —— 异步实现
  （如 httpx 连接池）用完要能关。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from .types import RerankDocument, RerankHit


class Reranker(ABC):
    """重排序的统一接口。"""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: Sequence[RerankDocument],
        *,
        top_n: int | None = None,
    ) -> list[RerankHit]:
        """按 `query` 重排 `documents`，返回按相关度降序的命中（最相关在前）。"""

    @abstractmethod
    async def aclose(self) -> None:
        """释放底层资源（连接池等）。幂等；之后再调用要能正常工作。"""

    async def __aenter__(self) -> Reranker:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
