"""own_bm25 的写入路径：insert / delete / replace / rebuild 与事务边界。

写入的验收标准是**索引与统计量始终一致**：`N`、`total_dl`、每词 `df`、每篇
`tf`/`dl` 四者任意一个错位都会让打分静默出错，所以每条用例都直接查表核对，
不只看 `search` 的表象。

事务边界按三层验：整批（中途失败整批回滚）、外层（调用方的 savepoint 能连本表
一起回滚）、跨连接（提交后的下一个连接看得到）。
"""

from __future__ import annotations

import sqlite3

import pytest

from simple_rag.db import Database
from simple_rag.repository.bm25 import create_bm25
from simple_rag.repository.bm25.own_bm25._schema import STATS_TABLE


def state(conn):
    """`(stats, {chunk_id: dl}, {chunk_id: {term: tf}})` —— 直接读表，不看 search。"""
    stats = conn.execute(f"select n, total_dl from {STATS_TABLE}").fetchone()
    docs = dict(
        conn.execute("select chunk_id, dl from own_bm25_docs").fetchall()
    )
    tf: dict[str, dict[str, int]] = {}
    rows = conn.execute(
        "select d.chunk_id, p.term, p.tf from own_bm25_postings p "
        "join own_bm25_docs d on d.doc_id = p.doc_id"
    )
    for chunk_id, term, count in rows:
        tf.setdefault(chunk_id, {})[term] = count
    return stats, docs, tf


def empty_state():
    return (0, 0), {}, {}


# ---------------------------------------------------------------- insert


def test_insert_写入并更新统计(conn, tok, index):
    assert index.insert([{"chunk_id": "a", "text": "x x y"}]) == 1
    assert index.insert([{"chunk_id": "b", "text": "y"}]) == 1
    assert state(conn) == ((2, 4), {"a": 3, "b": 1}, {"a": {"x": 2, "y": 1}, "b": {"y": 1}})


def test_insert_空文本计入_N_但不产生词频(conn, tok, index):
    index.insert([{"chunk_id": "a", "text": ""}])
    assert state(conn) == ((1, 0), {"a": 0}, {})
    assert index.search("x", k=5) == []


def test_insert_空批不动统计(index):
    assert index.insert([]) == 0


def test_insert_已存在的_id_抛_IntegrityError_并整批回滚(conn, tok, index):
    index.insert([{"chunk_id": "a", "text": "x"}])
    with pytest.raises(sqlite3.IntegrityError):
        index.insert([{"chunk_id": "b", "text": "y"}, {"chunk_id": "a", "text": "z"}])
    # 同一批里的 b 也不能留下
    assert state(conn) == ((1, 1), {"a": 1}, {"a": {"x": 1}})


def test_insert_批内重复抛_ValueError_并不落任何数据(conn, tok, index):
    with pytest.raises(ValueError, match="批内 chunk_id 重复"):
        index.insert([{"chunk_id": "a", "text": "x"}, {"chunk_id": "a", "text": "y"}])
    assert state(conn) == empty_state()


@pytest.mark.parametrize(
    "bad",
    [
        {"chunk_id": "a"},  # 缺 text
        {"chunk_id": 1, "text": "x"},  # chunk_id 不是 str
        {"chunk_id": "a", "text": "x", "extra": 1},  # 多余键
        "不是 mapping",
    ],
)
def test_insert_坏记录整批回滚(conn, tok, index, bad):
    def records():
        yield {"chunk_id": "ok", "text": "x"}
        yield bad

    with pytest.raises((TypeError, ValueError)):
        index.insert(records())
    assert state(conn) == empty_state()


def test_insert_后半程抛异常的iterable整批回滚(conn, tok, index):
    def records():
        yield {"chunk_id": "ok", "text": "x"}
        raise RuntimeError("迭代器故障（测试用）")

    with pytest.raises(RuntimeError):
        index.insert(records())
    assert state(conn) == empty_state()


def test_insert_分词器故障整批回滚(conn, broken_tokenizer):
    index = create_bm25(
        "own_bm25", tokenizer=broken_tokenizer(fail_from=1), conn=conn
    )
    index.open()
    with pytest.raises(RuntimeError, match="分词器故障"):
        index.insert([{"chunk_id": "a", "text": "x"}, {"chunk_id": "b", "text": "y"}])
    assert state(conn) == empty_state()


def test_失败后下一次调用正常(conn, tok, index):
    """TEMP 去重表在每次方法开始时清空 —— 上次失败留下的痕迹不能污染下次。"""
    with pytest.raises(ValueError, match="批内 chunk_id 重复"):
        index.insert([{"chunk_id": "a", "text": "x"}, {"chunk_id": "a", "text": "y"}])
    assert index.insert([{"chunk_id": "a", "text": "x"}]) == 1


# ---------------------------------------------------------------- delete


def test_delete_不存在或空批返回0(conn, tok, index):
    assert index.delete([]) == 0
    assert index.delete(["nope"]) == 0
    assert state(conn) == empty_state()


def test_delete_删词频与映射并回收统计(conn, tok, index):
    index.rebuild([{"chunk_id": "a", "text": "x y"}, {"chunk_id": "b", "text": "y z"}])
    assert index.delete(["a"]) == 1
    assert state(conn) == ((1, 2), {"b": 2}, {"b": {"y": 1, "z": 1}})
    assert index.search("x", k=5) == []


def test_delete_重复_id只算首次(conn, tok, index):
    index.insert([{"chunk_id": "a", "text": "x"}])
    assert index.delete(["a", "a", "a"]) == 1
    assert state(conn) == empty_state()


def test_delete_非字符串_id_报错且不落任何删除(conn, tok, index):
    index.insert([{"chunk_id": "a", "text": "x"}])
    with pytest.raises(TypeError, match="chunk_id 必须是 str"):
        index.delete(["a", 123])
    assert state(conn) == ((1, 1), {"a": 1}, {"a": {"x": 1}})


# ---------------------------------------------------------------- replace


def test_replace_是_upsert(conn, tok, index):
    assert index.replace([{"chunk_id": "a", "text": "x"}]) == 1
    assert state(conn) == ((1, 1), {"a": 1}, {"a": {"x": 1}})
    assert index.replace([{"chunk_id": "a", "text": "y y"}]) == 1
    # 旧的 x 必须消失，统计量按差值回收（不是 +1 累加）
    assert state(conn) == ((1, 2), {"a": 2}, {"a": {"y": 2}})


def test_replace_空文本覆盖旧词频(conn, tok, index):
    """改短到空 —— 旧词频不删干净的话会留下幽灵命中。"""
    index.replace([{"chunk_id": "a", "text": "x"}])
    index.replace([{"chunk_id": "a", "text": ""}])
    assert state(conn) == ((1, 0), {"a": 0}, {})
    assert index.search("x", k=5) == []


def test_replace_批内重复报错并回滚(conn, tok, index):
    index.replace([{"chunk_id": "keep", "text": "x"}])
    with pytest.raises(ValueError, match="批内 chunk_id 重复"):
        index.replace([{"chunk_id": "keep", "text": "y"}, {"chunk_id": "keep", "text": "z"}])
    assert state(conn) == ((1, 1), {"keep": 1}, {"keep": {"x": 1}})


def test_replace_空批返回0(index):
    assert index.replace([]) == 0


# ---------------------------------------------------------------- rebuild


def test_rebuild_丢弃旧数据并重置统计(conn, tok, index):
    index.rebuild([{"chunk_id": "a", "text": "x y"}, {"chunk_id": "b", "text": "z"}])
    assert index.rebuild([{"chunk_id": "c", "text": "w"}]) == 1
    assert state(conn) == ((1, 1), {"c": 1}, {"c": {"w": 1}})


def test_rebuild_空批清空(conn, tok, index):
    index.rebuild([{"chunk_id": "a", "text": "x"}])
    assert index.rebuild([]) == 0
    assert state(conn) == empty_state()


def test_rebuild_失败恢复旧索引(conn, tok, index):
    index.rebuild([{"chunk_id": "old", "text": "x"}])
    with pytest.raises(ValueError, match="批内 chunk_id 重复"):
        index.rebuild([{"chunk_id": "new", "text": "y"}, {"chunk_id": "new", "text": "z"}])
    assert state(conn) == ((1, 1), {"old": 1}, {"old": {"x": 1}})
    assert [h.chunk_id for h in index.search("x", k=5)] == ["old"]


# ---------------------------------------------------------------- 事务边界


def test_外层savepoint回滚带走本表与其他表(conn, tok, index):
    index.insert([{"chunk_id": "a", "text": "x"}])
    before = state(conn)
    conn.execute("create table other (v text)")

    conn.execute("savepoint outer")
    index.insert([{"chunk_id": "b", "text": "y"}])
    conn.execute("insert into other values ('v')")
    conn.execute("rollback to outer")
    conn.execute("release outer")

    assert state(conn) == before
    assert conn.execute("select count(*) from other").fetchone() == (0,)


def test_外层savepoint提交后本表可见(conn, tok, index):
    conn.execute("savepoint outer")
    index.insert([{"chunk_id": "a", "text": "x"}])
    conn.execute("release outer")
    assert state(conn) == ((1, 1), {"a": 1}, {"a": {"x": 1}})


def test_跨连接可见新统计(tmp_path, tok):
    path = tmp_path / "shared.db"
    writer = Database(path)
    reader = Database(path)
    try:
        w = create_bm25("own_bm25", tokenizer=tok, conn=writer.connection)
        w.open()
        r = create_bm25("own_bm25", tokenizer=tok, conn=reader.connection)
        r.open()
        w.rebuild([{"chunk_id": "a", "text": "x x"}])
        assert [h.chunk_id for h in r.search("x", k=5)] == ["a"]
        w.delete(["a"])
        assert r.search("x", k=5) == []
    finally:
        writer.close()
        reader.close()


def test_多实例不同参数不互相覆盖(conn, tok):
    """k1/b 只存在实例里，不进库 —— 同一个连接上换参数不用重建。"""
    default = create_bm25("own_bm25", tokenizer=tok, conn=conn)
    default.open()
    default.rebuild([{"chunk_id": "long", "text": "x x x x"}, {"chunk_id": "short", "text": "x"}])

    flat = create_bm25("own_bm25", tokenizer=tok, conn=conn, k1=0, b=0)
    flat.open()
    assert default.search("x", k=5)[0].chunk_id == "long"
    assert flat.search("x", k=5)[0].chunk_id == "long"  # k1=0 -> 同分，按 chunk_id
    assert default.search("x", k=5)[0].score != flat.search("x", k=5)[0].score
    # 默认参数的实例不受影响
    assert default.search("x", k=5)[0].score != flat.search("x", k=5)[0].score


def test_统计行被删后报错而不是静默空结果(conn, tok, index):
    index.rebuild([{"chunk_id": "a", "text": "x"}])
    conn.execute(f"delete from {STATS_TABLE}")
    with pytest.raises(ValueError, match="绕过接口"):
        index.search("x", k=5)


@pytest.mark.parametrize(
    "call",
    [
        lambda index: index.insert([{"chunk_id": "a", "text": "x"}]),
        lambda index: index.replace([{"chunk_id": "a", "text": "x"}]),
        lambda index: index.delete(["keep"]),
        lambda index: index.rebuild([{"chunk_id": "a", "text": "x"}]),
    ],
)
def test_统计行被删后写路径也报错(conn, tok, index, call):
    """写路径不能静默 no-op —— 返回「写入成功」而统计量不动，比报错糟糕得多。

    读路径（`search`）与写路径（`bump_stats` / `reset_stats`）姿态必须一致。
    """
    index.rebuild([{"chunk_id": "keep", "text": "x"}])
    conn.execute(f"delete from {STATS_TABLE}")
    with pytest.raises(ValueError, match="绕过接口"):
        call(index)


def test_统计行还在时_delete_空批不写统计量(index):
    """对照组：空批不碰统计量，所以它不该被上面那条检查误伤。"""
    index.rebuild([{"chunk_id": "a", "text": "x"}])
    assert index.delete([]) == 0
