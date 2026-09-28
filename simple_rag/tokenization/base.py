"""分词器的注册表与共用逻辑（协议 + 纯函数，不碰数据库）。

**为什么单独一层**：分词器与索引绑定 —— 换了分词器，已落盘的 token 串全部失效、
必须全库重建。它既不属于 `bm25`，也不属于将来的任何一个消费者，所以单独成包。

**注册表放在这里而不是 `__init__.py`**：`jieba_tokenizer.py` 要 `from .base import ...`，
注册表若在 `__init__.py` 就会形成 `__init__ → jieba_tokenizer → __init__` 的循环。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """分词器契约。

    实现必须保证两件事，否则 token 串还原会错位：

    1. 返回的 token **不含纯空白** —— jieba 会把空白本身切成独立 token
       （`'a b'` → `['a', ' ', 'b']`，制表符 / 换行 / 全角空格同理，实测）。
    2. 返回的 token **不含内部空格** —— token 串用空格拼接落盘，含空格会破坏还原。

    第 1 条用 `drop_blank` 保证。

    ⚠️ **换分词器 = 换一套 token，必须全库重建。** 写入侧与查询侧用了不同的
    分词器（含同分词器不同版本）时，同一个词在两侧不是同一个 token ——
    **结果是静默零召回，不报错。**
    """

    name: str

    def tokenize(self, text: str) -> list[str]:
        """把原文切成 token 序列。"""
        ...


def drop_blank(tokens: Iterable[str]) -> list[str]:
    """丢掉纯空白 token。

    实测：`jieba.lcut('hello world')` → `['hello', ' ', 'world']`；
    制表符 / 换行 / 全角空格同样被切成独立 token。
    """
    return [t for t in tokens if t.strip()]


def to_document(tokens: Iterable[str]) -> str:
    """token 序列 → 落盘用的 token 串（空格拼接）。

    写入侧与查询侧共用这一个函数 —— 两端格式不一致就是静默零召回。
    """
    return " ".join(tokens)


_TOKENIZERS: dict[str, Callable[[], Tokenizer]] = {}


def register_tokenizer(name: str) -> Callable[[type], type]:
    """类装饰器：把分词器注册进注册表。

    重复注册同一个名字报 `ValueError` —— 静默覆盖会让「配置到底选了哪个」
    变得不可知。

    用法::

        @register_tokenizer("pkuseg")
        class PkusegTokenizer:
            name = "pkuseg"
            def tokenize(self, text: str) -> list[str]: ...
    """

    def decorator(cls: type) -> type:
        if name in _TOKENIZERS:
            raise ValueError(
                f"分词器名字 {name!r} 已经注册过了，"
                f"已注册的是 {TokenizerFactory.available()}"
            )
        _TOKENIZERS[name] = cls
        return cls

    return decorator


class TokenizerFactory:
    """按名字建分词器 —— 可插拔分词器的唯一入口。"""

    @staticmethod
    def create(name: str) -> Tokenizer:
        """建一个分词器。未注册的名字报 `ValueError`，消息里列出可用的。"""
        factory = _TOKENIZERS.get(name)
        if factory is None:
            raise ValueError(
                f"没有注册名为 {name!r} 的分词器，"
                f"可用的是 {TokenizerFactory.available()}"
            )
        return factory()

    @staticmethod
    def available() -> tuple[str, ...]:
        """已注册的分词器名字，排序后返回。"""
        return tuple(sorted(_TOKENIZERS))

    @staticmethod
    def registered(name: str) -> bool:
        return name in _TOKENIZERS
