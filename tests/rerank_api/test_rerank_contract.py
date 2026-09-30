"""`rerank_api` 统一接口的契约测试。

钉死三件事：接口不可绕过（抽象）、实现必须满足（is-a + 生命周期）、
实现的短路行为与接口语义一致（空输入不发请求）。
不打真网络 —— HTTP 契约由 `explore/rerank/` 的探针负责。

文件名带 `rerank_` 前缀：tests 目录没有 `__init__.py`，pytest 按文件名导入，
全仓库里测试文件名不能撞（bm25 已有 `test_contract.py`）。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from simple_rag.rerank_api import AliyunReranker, Reranker
from simple_rag.rerank_api.aliyun import AliyunRerankAPIConfig


def test_reranker_is_abstract() -> None:
    """统一接口本身不可实例化 —— 调用方必须选一个实现。"""
    with pytest.raises(TypeError):
        Reranker()  # type: ignore[abstract]


def test_aliyun_reranker_is_a_reranker() -> None:
    assert isinstance(AliyunReranker(AliyunRerankAPIConfig()), Reranker)


def test_base_provides_async_context_manager() -> None:
    """生命周期是接口契约的一部分：任意实现挂上 base 就获得 `async with`，
    退出时必须走到 `aclose()` —— 厂商实现自己不用重复写这对方法。"""

    closed: list[bool] = []

    class DummyReranker(Reranker):
        async def rerank(self, query, documents, *, top_n=None):
            return []

        async def aclose(self) -> None:
            closed.append(True)

    async def run() -> None:
        async with DummyReranker():
            assert not closed
        assert closed == [True], "async with 退出必须调用 aclose"

        # aclose 幂等：再关两次不炸
        dummy = DummyReranker()
        await dummy.aclose()
        await dummy.aclose()

    asyncio.run(run())


def test_aliyun_config_validation() -> None:
    """缺字段的报错在构造期，不等第一次请求才炸。"""
    for missing in ("api_base", "api_key", "model"):
        config = SimpleNamespace(api_base="https://x", api_key="k", model="m")
        setattr(config, missing, "")
        with pytest.raises(ValueError, match=missing):
            AliyunReranker(config)


def test_aliyun_empty_documents_short_circuit() -> None:
    """接口语义：documents 为空返回 []，不发请求（不会碰配置里的地址）。"""
    config = SimpleNamespace(api_base="https://invalid.test", api_key="k", model="m")
    reranker = AliyunReranker(config)
    assert asyncio.run(reranker.rerank("q", [])) == []
