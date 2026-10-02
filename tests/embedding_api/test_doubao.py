"""`DoubaoEmbeddingVision` 的 HTTP 交互测试 —— 全部走 `httpx.MockTransport`，不打真网络。

mock 的只是 HTTP 层（外部 API）；请求体组装、响应解析、稀疏向量的前置校验
这些被测核心逻辑**不 mock**。真接口的端到端契约由 `explore/实际测试/` 的脚本负责。

响应形状有两种 —— 251215 是对象，更早版本是数组，`_extract_result` 两种都兼容，
所以两种都要测。
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from simple_rag.config import DoubaoAPIConfig
from simple_rag.embedding_api import DoubaoEmbeddingVision, ImagePart, TextPart


class FakeAPI:
    """假豆包端点：记录收到的请求，按脚本返回响应。

    `statuses` 给了就按序返回（用完停在最后一个），没给就一律返回 `status`。
    """

    def __init__(
        self, payload: dict, *, status: int = 200, statuses: list[int] | None = None
    ) -> None:
        self.requests: list[dict] = []  # 每次请求的 {url, authorization, body}
        self.payload = payload
        self.status = status
        self.statuses = statuses

    def _status_for(self, index: int) -> int:
        if self.statuses is None:
            return self.status
        return self.statuses[min(index, len(self.statuses) - 1)]

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(
            {
                "url": str(request.url),
                "authorization": request.headers.get("authorization"),
                "body": json.loads(request.content),
            }
        )
        status = self._status_for(len(self.requests) - 1)
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "bad"}})
        return httpx.Response(200, json=self.payload)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """把 `httpx.AsyncClient` 换成注入 MockTransport 的工厂。

        `DoubaoEmbeddingVision` 内部按模块属性取 `httpx.AsyncClient(timeout=...)`，
        patch 模块属性即可生效；monkeypatch 自动还原。
        """
        handler = self.handle
        real_cls = httpx.AsyncClient

        def factory(**kwargs: object) -> httpx.AsyncClient:
            kwargs["transport"] = httpx.MockTransport(handler)
            return real_cls(**kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(httpx, "AsyncClient", factory)


def make_client(
    *, api_base: str = "https://fake.test", **kwargs: object
) -> DoubaoEmbeddingVision:
    return DoubaoEmbeddingVision(
        DoubaoAPIConfig(api_base=api_base, api_key="k-test", model="m-test"),
        **kwargs,  # type: ignore[arg-type]
    )


def ok_payload(dense: list[float], sparse: list[dict] | None = None) -> dict:
    """造一个成功的响应体。`sparse` 给了才带 `sparse_embedding` 字段。"""
    data: dict = {"embedding": dense}
    if sparse is not None:
        data["sparse_embedding"] = sparse
    return {"data": data}


# ---------------------------------------------------------------- 请求组装


def test_请求体形状(monkeypatch: pytest.MonkeyPatch) -> None:
    """契约：请求体 = {model, input, encoding_format}，可选字段不传就不出现。

    验证方式：
    1. 假端点记录请求体
    2. embed([TextPart("你好")])
    3. 断言 model / input 的形状 / encoding_format，且**不含** instructions、
       dimensions、sparse_embedding（上游按「不传 = 服务端默认」处理）

    如果失败，说明请求体组装偏离了接口语义。
    """
    api = FakeAPI(ok_payload([0.1, 0.2]))
    api.install(monkeypatch)

    asyncio.run(make_client().embed([TextPart("你好")]))

    body = api.requests[0]["body"]
    assert body["model"] == "m-test"
    assert body["input"] == [{"type": "text", "text": "你好"}]
    assert body["encoding_format"] == "float"
    assert "instructions" not in body
    assert "dimensions" not in body
    assert "sparse_embedding" not in body


def test_多模态片段按类型转payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """文本 / 图片 / 视频三种片段各自的 payload 形状不同。"""
    api = FakeAPI(ok_payload([0.1]))
    api.install(monkeypatch)

    asyncio.run(
        make_client().embed(
            [
                TextPart("描述这张图"),
                ImagePart("https://img.test/a.png"),
                ImagePart("data:image/png;base64,AAAA"),
            ]
        )
    )

    assert api.requests[0]["body"]["input"] == [
        {"type": "text", "text": "描述这张图"},
        {"type": "image_url", "image_url": {"url": "https://img.test/a.png"}},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]


def test_可选字段按需出现(monkeypatch: pytest.MonkeyPatch) -> None:
    """instructions / dimensions / sparse_embedding 传了才出现，值原样透传。"""
    api = FakeAPI(ok_payload([0.1], sparse=[{"index": 3, "value": 0.5}]))
    api.install(monkeypatch)

    asyncio.run(
        make_client().embed(
            [TextPart("你好")],
            instructions="给检索用",
            dimensions=1024,
            sparse=True,
        )
    )

    body = api.requests[0]["body"]
    assert body["instructions"] == "给检索用"
    assert body["dimensions"] == 1024
    assert body["sparse_embedding"] == {"type": "enabled"}


def test_端点是api_base拼embeddings_multimodal(monkeypatch: pytest.MonkeyPatch) -> None:
    """`api_base` 是 host，路径由本类拼；结尾多余的 `/` 不产生双斜杠。"""
    api = FakeAPI(ok_payload([0.1]))
    api.install(monkeypatch)

    asyncio.run(make_client(api_base="https://fake.test/").embed([TextPart("x")]))

    assert api.requests[0]["url"] == "https://fake.test/embeddings/multimodal"
    assert api.requests[0]["authorization"] == "Bearer k-test"


# ---------------------------------------------------------------- 前置校验


def test_稀疏向量只支持纯文本_带图片时拦截且不发请求(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """契约：`sparse=True` + 非文本片段 → `ValueError`，**且不发请求**。

    服务端对「稀疏 + 图片/视频」直接 400；本地拦下可以省一次无效往返。
    """
    api = FakeAPI(ok_payload([0.1]))
    api.install(monkeypatch)

    with pytest.raises(ValueError, match="稀疏向量只支持纯文本"):
        asyncio.run(
            make_client().embed([TextPart("文字"), ImagePart("u")], sparse=True)
        )

    assert api.requests == []


def test_稀疏向量配纯文本放行(monkeypatch: pytest.MonkeyPatch) -> None:
    """纯文本 + `sparse=True` 是合法的，不该被上面的校验误伤。"""
    api = FakeAPI(ok_payload([0.1], sparse=[]))
    api.install(monkeypatch)

    result = asyncio.run(
        make_client().embed([TextPart("a"), TextPart("b")], sparse=True)
    )

    assert result.sparse == []
    assert len(api.requests) == 1


# ---------------------------------------------------------------- 响应解析


def test_响应data是对象时解析(monkeypatch: pytest.MonkeyPatch) -> None:
    """新模型版本（251215）：`data` 是对象。"""
    api = FakeAPI(ok_payload([0.1, 0.2, 0.3]))
    api.install(monkeypatch)

    result = asyncio.run(make_client().embed([TextPart("x")]))

    assert result.dense == [0.1, 0.2, 0.3]
    assert result.sparse is None


def test_响应data是数组时解析(monkeypatch: pytest.MonkeyPatch) -> None:
    """更早版本：`data` 是数组。多模态接口把片段融成一个向量，所以只有一个元素。"""
    api = FakeAPI({"data": [{"embedding": [0.4, 0.5]}]})
    api.install(monkeypatch)

    result = asyncio.run(make_client().embed([TextPart("x")]))

    assert result.dense == [0.4, 0.5]


def test_稀疏向量解析成SparseEntry(monkeypatch: pytest.MonkeyPatch) -> None:
    """`sparse_embedding` 的每项转成 `SparseEntry(index, value)`，顺序不变。"""
    api = FakeAPI(
        ok_payload([0.1], sparse=[{"index": 7, "value": 0.25}, {"index": 2, "value": 0.5}])
    )
    api.install(monkeypatch)

    result = asyncio.run(make_client().embed([TextPart("x")], sparse=True))

    assert result.sparse is not None
    assert [(e.index, e.value) for e in result.sparse] == [(7, 0.25), (2, 0.5)]


# ---------------------------------------------------------------- 错误与生命周期


def test_非2xx抛HTTPStatusError(monkeypatch: pytest.MonkeyPatch) -> None:
    """契约：HTTP 非 2xx → 抛 `httpx.HTTPStatusError`，不许静默返回空向量。"""
    api = FakeAPI({}, status=401)
    api.install(monkeypatch)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        asyncio.run(make_client().embed([TextPart("x")]))

    assert exc_info.value.response.status_code == 401


# ---------------------------------------------------------------- 429 退避重试


class _Slept(list):
    """记录每次退避实际等了多久。"""


@pytest.fixture
def slept(monkeypatch: pytest.MonkeyPatch) -> _Slept:
    """把 `asyncio.sleep` 换掉 —— 真等 8 秒的测试没人跑。

    换的是模块属性，`doubao` 里写 `asyncio.sleep(...)` 所以会走到这里。
    """
    recorded = _Slept()

    async def fake_sleep(seconds: float) -> None:
        recorded.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return recorded


def test_429重试后成功(monkeypatch: pytest.MonkeyPatch, slept: _Slept) -> None:
    """限流是暂时的：退避之后重试，最终拿到正常结果。

    实测（2026-10-02）：豆包把限流报成 429 `AccountRateLimitExceeded`，
    **不返回 `retry-after`** —— 节奏只能客户端自己定，所以必须自己退避。
    """
    api = FakeAPI(ok_payload([0.1]), statuses=[429, 429, 200])
    api.install(monkeypatch)

    result = asyncio.run(make_client().embed([TextPart("x")]))

    assert result.dense == [0.1]
    assert len(api.requests) == 3


def test_退避是翻倍的(monkeypatch: pytest.MonkeyPatch, slept: _Slept) -> None:
    """等的时间按 2 的幂增长，不是固定间隔。"""
    api = FakeAPI(ok_payload([0.1]), statuses=[429, 429, 200])
    api.install(monkeypatch)

    asyncio.run(make_client(retry_base_wait=1.0).embed([TextPart("x")]))

    assert slept == [1.0, 2.0]


def test_重试次数用尽后抛出(monkeypatch: pytest.MonkeyPatch, slept: _Slept) -> None:
    """一直 429 就抛 —— 不许无限重试，也不许静默返回空向量。"""
    api = FakeAPI(ok_payload([0.1]), statuses=[429])
    api.install(monkeypatch)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        asyncio.run(make_client().embed([TextPart("x")]))

    assert exc_info.value.response.status_code == 429
    assert len(api.requests) == 4  # 首发 1 次 + 默认重试 3 次


def test_重试次数可配置(monkeypatch: pytest.MonkeyPatch, slept: _Slept) -> None:
    api = FakeAPI(ok_payload([0.1]), statuses=[429])
    api.install(monkeypatch)

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(make_client(max_retries=1).embed([TextPart("x")]))

    assert len(api.requests) == 2


def test_重试次数为0时一次都不重试(
    monkeypatch: pytest.MonkeyPatch, slept: _Slept
) -> None:
    api = FakeAPI(ok_payload([0.1]), statuses=[429])
    api.install(monkeypatch)

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(make_client(max_retries=0).embed([TextPart("x")]))

    assert len(api.requests) == 1
    assert slept == []


def test_非429不重试(monkeypatch: pytest.MonkeyPatch, slept: _Slept) -> None:
    """400 是请求本身的问题，重试多少次都一样 —— 立刻抛。"""
    api = FakeAPI({}, status=400)
    api.install(monkeypatch)

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(make_client().embed([TextPart("x")]))

    assert len(api.requests) == 1
    assert slept == []


def test_aclose幂等(monkeypatch: pytest.MonkeyPatch) -> None:
    """契约：`aclose()` 可重复调用；关掉之后再 embed 要能重新建连接。"""
    api = FakeAPI(ok_payload([0.1]))
    api.install(monkeypatch)
    client = make_client()

    async def run() -> None:
        await client.embed([TextPart("x")])
        await client.aclose()
        await client.aclose()  # 第二次不该抛
        await client.embed([TextPart("x")])  # 关掉后懒重建

    asyncio.run(run())

    assert len(api.requests) == 2


def test_async_with自动关闭(monkeypatch: pytest.MonkeyPatch) -> None:
    """`async with` 退出时自动 `aclose()`，语义与显式调用一致。"""
    api = FakeAPI(ok_payload([0.1]))
    api.install(monkeypatch)

    async def run() -> list[float]:
        async with make_client() as client:
            return (await client.embed([TextPart("x")])).dense

    assert asyncio.run(run()) == [0.1]


@pytest.mark.parametrize(
    "kwargs, missing",
    [
        ({"api_key": "k", "model": "m"}, "api_base"),
        ({"api_base": "u", "model": "m"}, "api_key"),
        ({"api_base": "u", "api_key": "k"}, "model"),
    ],
)
def test_缺配置构造即报错(
    kwargs: dict, missing: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """三个字段缺任一都在构造时报错，不留到发请求时才炸。

    必须清掉环境变量：`simple_rag.config` 在 import 时把项目根的 `.env`
    读进了 `os.environ`，不清的话缺的字段会被环境补齐，用例就测不到东西。
    """
    for var in ("DOUBAO_API_BASE", "DOUBAO_API_KEY", "DOUBAO_EMBEDDING_MODEL_ID"):
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(ValueError, match=missing):
        DoubaoEmbeddingVision(DoubaoAPIConfig(**kwargs))
