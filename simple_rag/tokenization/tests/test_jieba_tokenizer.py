"""`JiebaTokenizer` 的测试 —— 素材全部来自 explore/bm25/FINDINGS.md §4 的实测。"""

from __future__ import annotations

import pytest

from simple_rag.tokenization import JiebaTokenizer, Tokenizer, to_document


@pytest.fixture(scope="module")
def tok() -> JiebaTokenizer:
    return JiebaTokenizer()


def test_name(tok):
    assert tok.name == "jieba"


def test_满足_Tokenizer_协议(tok):
    """`Tokenizer` 是 `runtime_checkable` 的 Protocol。"""
    assert isinstance(tok, Tokenizer)


def test_中文被切开而不是整句一个_token(tok):
    """这是整个 BM25 方案存在的理由。

    FTS5 的 `unicode61` 直接把整句当一个 token，查 `检索` 零命中
    （实测 explore/bm25/FINDINGS.md §4.1）。
    """
    tokens = tok.tokenize("向量检索是基于语义的检索方法")
    assert len(tokens) > 1
    assert "检索" in tokens


def test_空格被过滤(tok):
    """jieba 把空白切成独立 token —— 必须丢掉，否则 token 串还原会错位。"""
    assert " " not in tok.tokenize("hello world")


def test_制表符被过滤(tok):
    assert "\t" not in tok.tokenize("a\tb")


def test_换行被过滤(tok):
    assert "\n" not in tok.tokenize("中文\ntext")


def test_全角空格被过滤(tok):
    assert "　" not in tok.tokenize("全角　空格")


def test_空串返回空(tok):
    assert tok.tokenize("") == []


def test_纯空白返回空(tok):
    assert tok.tokenize("   \t\n　") == []


def test_token_不含内部空格(tok):
    """空格是落盘的拼接分隔符 —— token 内部含空格会破坏还原。"""
    tokens = tok.tokenize("hello world 向量 检索")
    assert all(" " not in t for t in tokens)


def test_to_document_往返稳定(tok):
    """写入侧拼出来的串，查询侧 `split()` 回来必须一模一样。"""
    tokens = tok.tokenize("向量检索是基于语义的检索方法")
    assert to_document(tokens).split() == tokens


def test_写入侧与查询侧切出同一个词(tok):
    """两侧一致是分词的唯一契约 —— 不一致就是静默零召回。"""
    doc_tokens = tok.tokenize("向量检索是基于语义的检索方法")
    query_tokens = tok.tokenize("检索")
    assert set(query_tokens) <= set(doc_tokens)


def test_可以重复构造(tok):
    """jieba 内部有 `initialized` 短路，重复构造不会重复加载词典。"""
    assert JiebaTokenizer().name == "jieba"
    assert JiebaTokenizer().name == "jieba"
