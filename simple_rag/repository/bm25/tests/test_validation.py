"""`base.validation` 的纯单测 —— 两个实现共用的校验，不碰任何存储。"""

from __future__ import annotations

import pytest

from simple_rag.repository.bm25.base.validation import (
    check_k,
    check_record,
    prepare_rows,
)

# 32 位 hex —— 和本项目 Chunk.chunk_id 的形状一致
CID = "0123456789abcdef0123456789abcdef"
OTHER = "fedcba9876543210fedcba9876543210"


# ---------------------------------------------------------------- k


def test_check_k_边界通过():
    assert check_k(0) == 0
    assert check_k(4096) == 4096


@pytest.mark.parametrize("bad", [4097, -1, True, "1", 1.0, None])
def test_check_k_非法报错(bad):
    with pytest.raises((TypeError, ValueError)):
        check_k(bad)


# ---------------------------------------------------------------- 记录


def test_check_record_返回元组():
    assert check_record({"chunk_id": CID, "text": "正文"}) == (CID, "正文")


def test_check_record_不要求_hex():
    """hex 是 fts5 的 rowid 要求，通用层只管形状 —— rank_bm25 用得上。"""
    assert check_record({"chunk_id": "任意id", "text": "正文"}) == ("任意id", "正文")


@pytest.mark.parametrize(
    "bad, exc",
    [
        ({"chunk_id": CID}, ValueError),
        ({"text": "正文"}, ValueError),
        ({"chunk_id": CID, "text": "正文", "extra": 1}, ValueError),
        ({"chunk_id": 123, "text": "正文"}, TypeError),
        ({"chunk_id": CID, "text": 123}, TypeError),
        ("不是 mapping", TypeError),
    ],
)
def test_check_record_非法报错(bad, exc):
    with pytest.raises(exc):
        check_record(bad)


def test_prepare_rows_正常():
    rows = prepare_rows(
        [{"chunk_id": CID, "text": "a"}, {"chunk_id": OTHER, "text": "b"}]
    )
    assert rows == [(CID, "a"), (OTHER, "b")]


def test_prepare_rows_批内重复报错():
    """两个实现都不允许批内重复 —— fts5 撞 rowid，rank_bm25 下标映射错位。"""
    with pytest.raises(ValueError, match="批内 chunk_id 重复"):
        prepare_rows([{"chunk_id": CID, "text": "a"}, {"chunk_id": CID, "text": "b"}])


def test_prepare_rows_空输入():
    assert prepare_rows([]) == []
