"""`base` 的纯单测 —— 不碰数据库。"""

from __future__ import annotations

import pytest

from simple_rag.tokenization import base
from simple_rag.tokenization.base import (
    TokenizerFactory,
    drop_blank,
    register_tokenizer,
    to_document,
)


# ---------------------------------------------------------------- drop_blank


def test_drop_blank_丢掉纯空白():
    assert drop_blank(["a", " ", "\t", "\n", "　", "b"]) == ["a", "b"]


def test_drop_blank_不改实词内容():
    """只判 `t.strip()` 是否为空，不动 token 本身。"""
    assert drop_blank(["hello world"]) == ["hello world"]


def test_drop_blank_空输入():
    assert drop_blank([]) == []


# ---------------------------------------------------------------- to_document


def test_to_document_空格拼接():
    assert to_document(["向量", "检索"]) == "向量 检索"


def test_to_document_与_split往返():
    tokens = ["向量", "检索", "方法"]
    assert to_document(tokens).split() == tokens


def test_to_document_空输入():
    assert to_document([]) == ""


# ---------------------------------------------------------------- registry


def test_available_包含内置的_jieba():
    assert "jieba" in TokenizerFactory.available()


def test_available_是排序的元组():
    got = TokenizerFactory.available()
    assert isinstance(got, tuple)
    assert list(got) == sorted(got)


def test_create_未注册时报错并列出可用的():
    with pytest.raises(ValueError, match="jieba"):
        TokenizerFactory.create("没有这个分词器")


def test_registered():
    assert TokenizerFactory.registered("jieba")
    assert not TokenizerFactory.registered("没有这个")


def test_注册后能创建(monkeypatch):
    """monkeypatch 换掉注册表，测试后自动恢复 —— 不污染其他用例。"""
    monkeypatch.setattr(base, "_TOKENIZERS", dict(base._TOKENIZERS))

    @register_tokenizer("dummy")
    class Dummy:
        name = "dummy"

        def tokenize(self, text: str) -> list[str]:
            return text.split()

    assert base.TokenizerFactory.registered("dummy")
    assert "dummy" in base.TokenizerFactory.available()
    assert base.TokenizerFactory.create("dummy").name == "dummy"


def test_重复注册同名报错(monkeypatch):
    """静默覆盖会让「配置到底选了哪个」变得不可知。"""
    monkeypatch.setattr(base, "_TOKENIZERS", dict(base._TOKENIZERS))

    @register_tokenizer("dup")
    class First:
        name = "dup"

        def tokenize(self, text: str) -> list[str]:
            return []

    with pytest.raises(ValueError, match="dup"):

        @register_tokenizer("dup")
        class Second:
            name = "dup"

            def tokenize(self, text: str) -> list[str]:
                return []
