"""own_bm25 测试的共用夹具。

**分词器用自己写的 split 版，不用 jieba** —— 打分的验收要的是**可手算**的
token 序列，jieba 切出什么词是它的事，混进来就分不清失败来自打分还是分词。
统一契约测试（`tests/repository/bm25/test_contract.py`）那边才用真 jieba。

评分 oracle 放在 `test_index.py` 里，不放这里 —— 测试目录不是包（没有
`__init__.py`），模块间相对 import 拿不到它，只有 fixtures 能跨文件共用。
"""

from __future__ import annotations

import sqlite3

import pytest

from simple_rag.repository.bm25 import create_bm25


class SplitTokenizer:
    """按空白切词。满足 `Tokenizer` 协议的两条约束（无纯空白 token、无内部空格）。"""

    name = "split"

    def tokenize(self, text: str) -> list[str]:
        return text.split()


class BrokenTokenizer:
    """切到一半就炸 —— 用来验证写入路径的整批回滚。"""

    name = "broken"

    def __init__(self, fail_from: int) -> None:
        self._fail_from = fail_from
        self._seen = 0

    def tokenize(self, text: str) -> list[str]:
        if self._seen >= self._fail_from:
            raise RuntimeError("分词器故障（测试用）")
        self._seen += 1
        return text.split()


@pytest.fixture
def tok():
    return SplitTokenizer()


@pytest.fixture
def conn():
    """裸连接：本实现只用标准 SQLite 能力，不需要 sqlite-vec。

    `isolation_level = None` 与 `simple_rag.db.Database` 一致 —— 事务由模块
    自己用 savepoint 显式控制。
    """
    connection = sqlite3.connect(":memory:")
    connection.isolation_level = None
    yield connection
    connection.close()


@pytest.fixture
def broken_tokenizer():
    """造一个「切到第 n 篇就炸」的分词器工厂 —— 验证写入路径的整批回滚。"""

    def make(fail_from: int = 1) -> BrokenTokenizer:
        return BrokenTokenizer(fail_from)

    return make


@pytest.fixture
def index(conn, tok):
    instance = create_bm25("own_bm25", tokenizer=tok, conn=conn)
    instance.open()
    return instance
