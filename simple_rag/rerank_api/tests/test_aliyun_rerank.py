"""`AliyunReranker` 的 HTTP 交互测试 —— 全部走 `httpx.MockTransport`，不打真网络。

mock 的只是 HTTP 层（外部 API）；`index → chunk_id` 映射、分数排序、请求体
组装这些被测核心逻辑**不 mock**。真接口的端到端契约由 `explore/rerank/`
的探针负责。

响应形状来自探针实测（`explore/rerank/README.md`）：
`{"output": {"results": [{"index": <原始下标>, "relevance_score": float}]}}`。
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from simple_rag.rerank_api import AliyunReranker, RerankDocument
from simple_rag.rerank_api.aliyun import AliyunRerankAPIConfig

DOCS = [
    RerankDocument(chunk_id="c0", text="文本零"),
    RerankDocument(chunk_id="c1", text="文本一"),
    RerankDocument(chunk_id="c2", text="文本二"),
]


class FakeAPI:
    """假阿里云 rerank 端点：记录收到的请求，按脚本返回响应。"""

    def __init__(
        self,
        results: list[dict],
        *,
        status: int = 200,
    ) -> None:
        self.requests: list[dict] = []  # 每次请求的 {url, headers, body}
        self.results = results
        self.status = status

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(
            {
                "url": str(request.url),
                "authorization": request.headers.get("authorization"),
                "body": json.loads(request.content),
            }
        )
        if self.status != 200:
            return httpx.Response(self.status, json={"code": "InvalidApiKey", "message": "bad key"})
        return httpx.Response(200, json={"output": {"results": self.results}, "usage": {}})

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """把 `httpx.AsyncClient` 换成注入 MockTransport 的工厂。

        `AliyunReranker` 内部按模块属性取 `httpx.AsyncClient(timeout=...)`，
        patch 模块属性即可生效；monkeypatch 自动还原。
        """
        handler = self.handle
        real_cls = httpx.AsyncClient

        def factory(**kwargs: object) -> httpx.AsyncClient:
            kwargs["transport"] = httpx.MockTransport(handler)
            return real_cls(**kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(httpx, "AsyncClient", factory)


def make_reranker() -> AliyunReranker:
    return AliyunReranker(AliyunRerankAPIConfig(api_base="https://fake.test/rerank", api_key="k-test", model="m-test"))


def test_rerank_request_body_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    契约：请求体形状 = {model, input: {query, documents}}，不带多余字段。

    验证方式：
    1. 假端点记录请求体
    2. rerank(query, 3 篇文档)
    3. 断言 body 的 model / input.query / input.documents（取 text）都正确；
       **不包含** return_documents（文本在手按 index 回取，不发它）；
       top_n 未传时**不包含** parameters（上游不传 = 全量）。

    如果失败，说明：请求体组装偏离了接口语义（多发了字段会让上游计费
    或行为变化）。
    """
    api = FakeAPI([{"index": 0, "relevance_score": 0.9}])
    api.install(monkeypatch)

    asyncio.run(make_reranker().rerank("查询词", DOCS))

    assert len(api.requests) == 1
    body = api.requests[0]["body"]
    assert body["model"] == "m-test"
    assert body["input"]["query"] == "查询词"
    assert body["input"]["documents"] == ["文本零", "文本一", "文本二"]
    assert "parameters" not in body
    assert "return_documents" not in json.dumps(body)


def test_rerank_sends_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    契约：鉴权走 `Authorization: Bearer <api_key>`。

    验证方式：
    1. 假端点记录 Authorization 头
    2. rerank 一次
    3. 断言头部的值。

    如果失败，说明：上游会 401，且密钥可能被放进了错误的位置（如 body）。
    """
    api = FakeAPI([{"index": 0, "relevance_score": 0.9}])
    api.install(monkeypatch)

    asyncio.run(make_reranker().rerank("q", DOCS))

    assert api.requests[0]["authorization"] == "Bearer k-test"


def test_rerank_top_n_becomes_parameter(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    契约：`top_n=2` → `parameters.top_n = 2`；`top_n=None` → 不发 parameters。

    验证方式：
    1. 分别以 top_n=2 与不传各打一次假端点
    2. 对比两次请求体。

    如果失败，说明：top_n 没有透传成上游的 parameters（截断会失效）。
    """
    api = FakeAPI([{"index": 0, "relevance_score": 0.9}])
    api.install(monkeypatch)

    asyncio.run(make_reranker().rerank("q", DOCS, top_n=2))
    asyncio.run(make_reranker().rerank("q", DOCS))

    with_top_n = api.requests[0]["body"]
    without = api.requests[1]["body"]
    assert with_top_n["parameters"] == {"top_n": 2}
    assert "parameters" not in without


def test_rerank_maps_index_back_to_chunk_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    契约：上游的 `index` 是**原始 documents 的下标**，输出必须按它回取
    对应的 `chunk_id` / `text`（上游会乱序返回，index ≠ 排名位次）。

    验证方式：
    1. 假端点按探针实测的形状返回乱序结果：index 顺序 [2, 0, 1]，
       且分数恰为降序（0.9 / 0.5 / 0.1）—— 排序不改变顺序，输出顺序
       纯粹反映 index 回取
    2. rerank
    3. 断言输出的 chunk_id 依次是 c2, c0, c1，text 也跟着 index 走。

    如果失败，说明：把 index 当成了排名位次或没有回取 —— 输出的归属
    就全错了（这是本实现最核心的映射）。
    """
    api = FakeAPI(
        [
            {"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.5},
            {"index": 1, "relevance_score": 0.1},
        ]
    )
    api.install(monkeypatch)

    hits = asyncio.run(make_reranker().rerank("q", DOCS))

    assert [h.chunk_id for h in hits] == ["c2", "c0", "c1"]
    assert [h.text for h in hits] == ["文本二", "文本零", "文本一"]


def test_rerank_sorts_by_score_descending(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    契约：输出按 relevance_score **降序**（最相关在前），不依赖上游返回顺序
    —— 排序由本类保证。

    验证方式：
    1. 假端点故意按「低分在前」的顺序返回
    2. rerank
    3. 断言输出顺序是高分在前。

    如果失败，说明：直接透传了上游顺序，方向契约（最相关在前）被破坏。
    """
    api = FakeAPI(
        [
            {"index": 1, "relevance_score": 0.1},
            {"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.5},
        ]
    )
    api.install(monkeypatch)

    hits = asyncio.run(make_reranker().rerank("q", DOCS))

    assert [h.chunk_id for h in hits] == ["c2", "c0", "c1"]


def test_rerank_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    契约：HTTP 非 2xx → 抛 `httpx.HTTPStatusError`，不许静默返回空结果
    （调用方必须能区分「没有命中」和「调用失败」）。

    验证方式：
    1. 假端点返回 401
    2. rerank 应抛异常，且状态码在异常里。

    如果失败，说明：吞了错误 —— 调用方会把故障当成「无相关文档」。
    """
    api = FakeAPI([], status=401)
    api.install(monkeypatch)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        asyncio.run(make_reranker().rerank("q", DOCS))
    assert exc_info.value.response.status_code == 401


def test_rerank_reuses_client_and_rebuilds_after_close(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    契约：实例可复用（client 只建一次）；`aclose()` 后再调用要能正常工作
    （懒创建重建），即 aclose 幂等且不把实例用废。

    验证方式：
    1. 同一实例连打三次：两次 rerank → aclose → 再一次 rerank
    2. 假端点应收到 3 个请求，且每次都成功。

    如果失败，说明：aclose 把实例关死了（后续调用报错/发不出请求），
    或每次调用都在重建连接池。
    """
    api = FakeAPI([{"index": 0, "relevance_score": 0.9}])
    api.install(monkeypatch)

    async def run() -> None:
        reranker = make_reranker()
        await reranker.rerank("q1", DOCS)
        await reranker.rerank("q2", DOCS)
        await reranker.aclose()
        await reranker.rerank("q3", DOCS)
        await reranker.aclose()

    asyncio.run(run())
    assert len(api.requests) == 3
    assert [r["body"]["input"]["query"] for r in api.requests] == ["q1", "q2", "q3"]
