"""豆包（火山方舟）多模态向量化 —— `doubao-embedding-vision`。

接口：`POST {api_base}/embeddings/multimodal`
文档：https://docs.volcengine.com/docs/82379/1409290

按服务商分文件：一个服务商一个模块，`doubao` 只管豆包。

**异步** —— `httpx.AsyncClient`，与 `rerank_api/aliyun.py` 同一套生命周期。
embed 在检索链路上是逐查询调用，挂起等响应而不占线程。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from simple_rag.config import DoubaoAPIConfig


@dataclass(frozen=True)
class TextPart:
    """文本片段。"""

    text: str


@dataclass(frozen=True)
class ImagePart:
    """图片片段。`url` 可以是图片 URL，也可以是 `data:image/...;base64,...`。"""

    url: str


@dataclass(frozen=True)
class VideoPart:
    """视频片段。`url` 可以是视频 URL，也可以是 Base64 编码。"""

    url: str


Part = TextPart | ImagePart | VideoPart


def _to_payload(part: Part) -> dict[str, Any]:
    """把片段转成接口要求的 `input` 元素。"""
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, ImagePart):
        return {"type": "image_url", "image_url": {"url": part.url}}
    return {"type": "video_url", "video_url": {"url": part.url}}


@dataclass(frozen=True)
class SparseEntry:
    """稀疏向量的一个非零项。"""

    index: int
    value: float


@dataclass(frozen=True)
class DoubaoEmbeddingResult:
    """一次向量化的结果。

    稠密向量（`dense`）一直有；稀疏向量（`sparse`）只有「纯文本输入 + 显式开启」
    才有，否则是 `None`。两者在响应里是 `data` 下的**不同字段**。
    """

    dense: list[float]
    sparse: list[SparseEntry] | None = None


def _extract_result(payload: dict[str, Any]) -> DoubaoEmbeddingResult:
    """从响应里取向量。

    `data` 的形状随模型版本不同：251215 是对象，更早版本是数组，两种都兼容。
    多模态接口把一次请求里的片段融合成**一个**向量，所以数组只会有一个元素。
    """
    data = payload["data"]
    if isinstance(data, list):
        data = data[0]
    sparse = data.get("sparse_embedding")
    return DoubaoEmbeddingResult(
        dense=data["embedding"],
        sparse=(
            [SparseEntry(entry["index"], entry["value"]) for entry in sparse]
            if sparse is not None
            else None
        ),
    )


class DoubaoEmbeddingVision:
    """豆包多模态向量化客户端。一个实例可复用，不做重试。

    异步生命周期：`httpx.AsyncClient` 懒创建（第一次调用时才建，此时必然已在
    event loop 里），用完 `aclose()`，或直接 `async with` 管理。
    """

    def __init__(
        self,
        config: DoubaoAPIConfig,
        *,
        timeout: float = 60.0,
    ) -> None:
        if not config.api_base:
            raise ValueError("缺少 api_base：构造 DoubaoAPIConfig 时传入，或设置环境变量 DOUBAO_API_BASE")
        if not config.api_key:
            raise ValueError("缺少 api_key：构造 DoubaoAPIConfig 时传入，或设置环境变量 DOUBAO_API_KEY")
        if not config.model:
            raise ValueError("缺少 model：构造 DoubaoAPIConfig 时传入，或设置环境变量 DOUBAO_EMBEDDING_MODEL_ID")
        self._config = config
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def embed(
        self,
        parts: Sequence[Part],
        *,
        instructions: str | None = None,
        dimensions: int | None = None,
        sparse: bool = False,
    ) -> DoubaoEmbeddingResult:
        """把一组多模态片段融合成一个稠密向量（可选再要一个稀疏向量）。

        - `parts` —— 片段列表，文本 / 图片 / 视频可混合
        - `instructions` —— 推理提示词；不传则由服务端按输入模态生成默认值
        - `dimensions` —— 稠密向量维度，可选 `1024` / `2048`；不传用服务端默认（2048）
        - `sparse` —— 是否同时要稀疏向量。**只支持纯文本输入**（服务端硬限制，
          带图片 / 视频会 400，这里提前拦下）
        """
        if sparse and any(not isinstance(part, TextPart) for part in parts):
            raise ValueError("稀疏向量只支持纯文本输入，parts 里不能有图片 / 视频")

        body: dict[str, Any] = {
            "model": self._config.model,
            "input": [_to_payload(part) for part in parts],
            "encoding_format": "float",
        }
        if instructions is not None:
            body["instructions"] = instructions
        if dimensions is not None:
            body["dimensions"] = dimensions
        if sparse:
            body["sparse_embedding"] = {"type": "enabled"}

        response = await self._get_client().post(
            f"{self._config.api_base.rstrip('/')}/embeddings/multimodal",
            headers={"Authorization": f"Bearer {self._config.api_key}"},
            json=body,
        )
        response.raise_for_status()
        return _extract_result(response.json())

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        """关闭底层连接池。懒创建的 client 用完要关；之后再调用会重新建。"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> DoubaoEmbeddingVision:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
