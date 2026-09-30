"""阿里云百炼（MaaS）rerank 重排序 —— `qwen3.7-text-rerank`。

接口：`POST {api_base}`（原生 URL 已是完整端点，直接 POST）
探针结论：`explore/rerank/README.md` —— `index` 是原始 `documents` 的下标、
`relevance_score` 越大越相关、`return_documents` 默认 false。

按服务商分文件：一个服务商一个模块，`aliyun` 只管阿里云。
本类实现统一接口 `Reranker`（见 `base/`）—— 厂商差异（分数方向、index 语义、
生命周期）全部挡在本模块内，调用方只看 `base.Reranker`。

与 `embedding_api/doubao.py` 的两点差异：

- **异步** —— `httpx.AsyncClient`（rerank 在检索链路上是逐查询调用，挂起等响应
  而不占线程）；`requests` 没有异步形态。
- **不透出分数** —— 输出只有「按相关度排好序的 `chunk_id` + 文本」；分数在内部
  排完序就丢弃（调用方拿排序就够了，方向统一是本类的职责）。
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from simple_rag.config import AliyunRerankAPIConfig

from .base import RerankDocument, RerankHit, Reranker


class AliyunReranker(Reranker):
    """阿里云百炼 rerank 客户端。一个实例可复用，不做重试。

    异步生命周期：`httpx.AsyncClient` 懒创建（第一次调用时才建，此时必然已在
    event loop 里），用完 `aclose()`，或直接 `async with` 管理。
    """

    def __init__(
        self,
        config: AliyunRerankAPIConfig,
        *,
        timeout: float = 60.0,
    ) -> None:
        if not config.api_base:
            raise ValueError("缺少 api_base：构造 AliyunRerankAPIConfig 时传入，或设置环境变量 ALY_RERANK_BASE_URL")
        if not config.api_key:
            raise ValueError("缺少 api_key：构造 AliyunRerankAPIConfig 时传入，或设置环境变量 ALY_API_KEY")
        if not config.model:
            raise ValueError("缺少 model：构造 AliyunRerankAPIConfig 时传入，或设置环境变量 ALY_RERANK_API_MODEL")
        self._config = config
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def rerank(
        self,
        query: str,
        documents: Sequence[RerankDocument],
        *,
        top_n: int | None = None,
    ) -> list[RerankHit]:
        """按 `query` 重排 `documents`，返回按相关度降序的命中。

        - `documents` 为空 → 返回 `[]`，不发请求
        - `top_n` 限定返回条数；`None` = 全量返回（上游行为：不传 `top_n` 就是全部）
        - 上游响应里每条结果的 `index` 是**原始 `documents` 数组的下标**，用它
          回取 `chunk_id`；再按 `relevance_score` **降序**显式排一遍 —— 排序
          由本类保证，不依赖上游返回顺序
        """
        if not documents:
            return []

        body: dict = {
            "model": self._config.model,
            "input": {"query": query, "documents": [doc.text for doc in documents]},
        }
        # 不发 return_documents：文本本就在手上，按 index 回取，省 token
        if top_n is not None:
            body["parameters"] = {"top_n": top_n}

        response = await self._get_client().post(
            self._config.api_base,
            headers={"Authorization": f"Bearer {self._config.api_key}"},
            json=body,
        )
        response.raise_for_status()
        results = response.json()["output"]["results"]
        results.sort(key=lambda item: item["relevance_score"], reverse=True)
        return [
            RerankHit(
                chunk_id=documents[item["index"]].chunk_id,
                text=documents[item["index"]].text,
            )
            for item in results
        ]

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        """关闭底层连接池。懒创建的 client 用完要关；之后再调用会重新建。"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
