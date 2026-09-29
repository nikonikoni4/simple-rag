"""`FTS5BM25Index` 的端到端测试 —— 真开临时库，真跑 FTS5。"""

from __future__ import annotations

import hashlib

import pytest

from simple_rag.db import Database
from simple_rag.repository.bm25 import BM25SearchResult
from simple_rag.repository.bm25.fts5_bm25 import FTS5BM25Index
from simple_rag.repository.vec import Schema, VecDB
from simple_rag.tokenization import TokenizerFactory

DOC = "向量检索是基于语义的检索方法"
OTHER = "BM25 是基于词频的检索方法"


def cid(n: int) -> str:
    """32 位 hex —— 用和 `Chunk.chunk_id` 一样的 blake2b 生成。

    ⚠️ 不能用 `f"{n:032x}"`：那样前 15 位全是 0，rowid 会撞成同一个
    （rowid 只取前 15 位 hex，这是本模块的已知限制）。
    """
    return hashlib.blake2b(str(n).encode("utf-8"), digest_size=16).hexdigest()


@pytest.fixture
def tok():
    return TokenizerFactory.create("jieba")


@pytest.fixture
def database(tmp_path):
    db = Database(tmp_path / "bm25.db")
    yield db
    db.close()


@pytest.fixture
def index(database, tok):
    instance = FTS5BM25Index(database.connection, tok)
    instance.open()
    return instance


# ---------------------------------------------------------------- 基本流程


def test_建表后能插入并查到(index):
    assert index.insert([{"chunk_id": cid(1), "text": DOC}]) == 1
    hits = index.search("检索", k=5)
    assert [h.chunk_id for h in hits] == [cid(1)]
    assert isinstance(hits[0], BM25SearchResult)


def test_分数是负数(index):
    """FTS5 的 `bm25()` 整体取负 —— 越小越相关。"""
    index.insert([{"chunk_id": cid(1), "text": DOC}])
    assert index.search("检索", k=1)[0].score < 0


def test_结果按分数升序(index):
    index.insert([
        {"chunk_id": cid(1), "text": "向量 检索 语义 方法 文档"},
        {"chunk_id": cid(2), "text": "检索 检索 检索 检索"},
    ])
    hits = index.search("检索", k=5)
    assert [h.score for h in hits] == sorted(h.score for h in hits)


def test_中文检索能召回(index):
    """**整个方案存在的理由。**

    FTS5 的 `unicode61` 直接在原文上建索引时，中文整句是一个 token，查 `检索`
    零命中（实测 explore/bm25/FINDINGS.md §4.1）。走分词后必须能召回。
    """
    index.insert([{"chunk_id": cid(1), "text": DOC}])
    assert [h.chunk_id for h in index.search("检索", k=5)] == [cid(1)]


def test_不相关的查不到(index):
    index.insert([{"chunk_id": cid(1), "text": DOC}])
    assert index.search("量子力学", k=5) == []


def test_返回条数受_k_限制(index):
    for i in range(5):
        index.insert([{"chunk_id": cid(i), "text": "检索 方法"}])
    assert len(index.search("检索", k=2)) == 2


# ---------------------------------------------------------------- 空查询


def test_k_为0直接返回空(index):
    index.insert([{"chunk_id": cid(1), "text": DOC}])
    assert index.search("检索", k=0) == []


@pytest.mark.parametrize("blank", ["", "   ", "\t\n", "　"])
def test_空查询直接返回空且不发_SQL(index, blank):
    """`match ''` 是 FTS5 语法错误 —— 必须在发 SQL 前挡住。"""
    index.insert([{"chunk_id": cid(1), "text": DOC}])

    seen: list[str] = []
    index._conn.set_trace_callback(seen.append)
    try:
        assert index.search(blank, k=5) == []
    finally:
        index._conn.set_trace_callback(None)
    assert not any("match" in sql.lower() for sql in seen)


# ---------------------------------------------------------------- 打开


def test_open_幂等(index):
    """每次开库都调是最常规的用法 —— 不能报 `table already exists`。"""
    index.open()
    index.open()


def test_open_撞普通表报错(database, tok):
    database.connection.execute("create table chunks_fts (a text, b text)")
    with pytest.raises(ValueError, match="不是 FTS5"):
        FTS5BM25Index(database.connection, tok).open()


def test_open_撞别的_tokenizer_报错(database, tok):
    """tokenizer 不同 = token 不同 = **静默零召回** —— 必须拦。"""
    database.connection.execute(
        "create virtual table chunks_fts using fts5("
        "tokens, chunk_id unindexed, tokenize='trigram')"
    )
    with pytest.raises(ValueError, match="tokenizer"):
        FTS5BM25Index(database.connection, tok).open()


# ---------------------------------------------------------------- 全量重建


def test_rebuild_旧数据全没了(index):
    index.insert([
        {"chunk_id": cid(1), "text": DOC},
        {"chunk_id": cid(2), "text": OTHER},
    ])
    assert index.rebuild([{"chunk_id": cid(3), "text": "全新的语料"}]) == 1
    assert index.search("检索", k=5) == []
    assert index.search("语料", k=5)[0].chunk_id == cid(3)


def test_rebuild_空批把表清空(index):
    """「重建为空」和「重建为这批」语义一致 —— 空批不是 no-op。"""
    index.insert([{"chunk_id": cid(1), "text": DOC}])
    assert index.rebuild([]) == 0
    assert index.search("检索", k=5) == []


def test_rebuild_是FTS5表且幂等可再写(index, database, tok):
    """rebuild 删的是本模块的 FTS5 表，重建后普通流程（open/insert）照常。"""
    index.rebuild([{"chunk_id": cid(1), "text": DOC}])
    index.open()  # 不报错 = 重建出来的表 DDL 仍一致
    assert index.insert([{"chunk_id": cid(2), "text": OTHER}]) == 1
    assert index.search("词频", k=5)[0].chunk_id == cid(2)


def test_rebuild_坏数据回滚_旧数据保留(index):
    index.insert([{"chunk_id": cid(1), "text": DOC}])
    with pytest.raises(ValueError, match="hex"):
        index.rebuild([
            {"chunk_id": cid(2), "text": OTHER},
            {"chunk_id": "c-2", "text": OTHER},
        ])
    # 校验在写库前抛，旧数据原样
    assert index.search("检索", k=5)[0].chunk_id == cid(1)


def test_rebuild_savepoint内部失败回滚_旧数据保留(index):
    """a / b 撞 rowid 对 —— 校验拦不住（chunk_id 不重复），失败发生在
    savepoint 内部的 executemany，走的是 rollback 分支而不是写库前校验。"""
    index.insert([{"chunk_id": cid(1), "text": DOC}])
    a = "0123456789abcde" + "0" * 17
    b = "0123456789abcde" + "1" * 17
    with pytest.raises(ValueError, match="rowid 冲突"):
        index.rebuild([
            {"chunk_id": cid(2), "text": OTHER},
            {"chunk_id": a, "text": OTHER},
            {"chunk_id": b, "text": OTHER},  # 和 a 撞 rowid
        ])
    # 回滚到 savepoint —— 旧数据原样，连表都没被删
    assert index.search("检索", k=5)[0].chunk_id == cid(1)


# ---------------------------------------------------------------- 覆盖或插入


def test_replace_覆盖已有(index):
    index.insert([{"chunk_id": cid(1), "text": DOC}])
    assert index.replace([{"chunk_id": cid(1), "text": OTHER}]) == 1
    assert index.search("语义", k=5) == []
    assert index.search("词频", k=5)[0].chunk_id == cid(1)


def test_replace_不存在也直接写入(index):
    assert index.replace([{"chunk_id": cid(1), "text": DOC}]) == 1
    assert index.search("检索", k=5)[0].chunk_id == cid(1)


def test_replace_同一批里新旧混合(index):
    index.insert([{"chunk_id": cid(1), "text": DOC}])
    assert index.replace([
        {"chunk_id": cid(1), "text": OTHER},          # 覆盖
        {"chunk_id": cid(2), "text": "第二条 语料"},   # 新插
    ]) == 2
    assert index.search("词频", k=5)[0].chunk_id == cid(1)
    assert index.search("语料", k=5)[0].chunk_id == cid(2)


def test_replace_失败整批回滚(index):
    """删+插被一个 savepoint 圈住 —— 插入失败时被删的旧行必须回来。

    a / b 是「不同 chunk_id、前 15 位 hex 相同」的 rowid 撞车对 —— 校验拦不住
    （批内 chunk_id 不重复），失败发生在 savepoint 内部的 executemany。
    撞 rowid 抛的 `IntegrityError` 被翻译成 `ValueError`，和 insert 一致。
    """
    index.insert([{"chunk_id": cid(1), "text": DOC}])
    a = "0123456789abcde" + "0" * 17
    b = "0123456789abcde" + "1" * 17
    with pytest.raises(ValueError, match="rowid 冲突"):
        index.replace([
            {"chunk_id": cid(1), "text": OTHER},
            {"chunk_id": a, "text": OTHER},
            {"chunk_id": b, "text": OTHER},  # 和 a 撞 rowid，插到这里炸
        ])
    # cid(1) 的旧行必须原样 —— 删了没回滚就是数据丢失
    assert index.search("语义", k=5)[0].chunk_id == cid(1)


def test_replace_空批返回0(index):
    assert index.replace([]) == 0


# ---------------------------------------------------------------- 删除


def test_删除后查不到(index):
    index.insert([{"chunk_id": cid(1), "text": DOC}])
    assert index.delete([cid(1)]) == 1
    assert index.search("检索", k=5) == []


def test_删不存在的返回0(index):
    assert index.delete([cid(99)]) == 0


def test_批量删(index):
    for i in range(5):
        index.insert([{"chunk_id": cid(i), "text": DOC}])
    assert index.delete([cid(i) for i in range(3)]) == 3
    assert len(index.search("检索", k=10)) == 2


def test_删空列表返回0(index):
    assert index.delete([]) == 0


def test_删完能重新插入(index):
    index.insert([{"chunk_id": cid(1), "text": DOC}])
    index.delete([cid(1)])
    index.insert([{"chunk_id": cid(1), "text": OTHER}])
    assert index.search("词频", k=5)[0].chunk_id == cid(1)


# ---------------------------------------------------------------- 异常


def test_插入非_hex_的_chunk_id_报错(index):
    with pytest.raises(ValueError, match="hex"):
        index.insert([{"chunk_id": "c-1", "text": DOC}])


def test_批内重复报错且一行没进(index):
    with pytest.raises(ValueError, match="批内 chunk_id 重复"):
        index.insert([
            {"chunk_id": cid(1), "text": DOC},
            {"chunk_id": cid(1), "text": OTHER},
        ])
    assert index.search("检索", k=5) == []


def test_text_非_str_报错(index):
    with pytest.raises(TypeError):
        index.insert([{"chunk_id": cid(1), "text": 123}])


def test_插空批次返回0(index):
    assert index.insert([]) == 0


def test_重复插入同一个_chunk_id_报_ValueError(index):
    """实测：FTS5 撞 rowid 抛的是 `IntegrityError("constraint failed")`，翻译掉。"""
    index.insert([{"chunk_id": cid(1), "text": DOC}])
    with pytest.raises(ValueError, match="rowid 冲突"):
        index.insert([{"chunk_id": cid(1), "text": OTHER}])


def test_前15位相同的两个_chunk_id_会报错(index):
    """60 bit 截断的后果 —— 第二条会盖掉第一条，所以必须报错而不是静默覆盖。"""
    a = "0123456789abcde" + "0" * 17
    b = "0123456789abcde" + "1" * 17
    assert a != b
    index.insert([{"chunk_id": a, "text": DOC}])
    with pytest.raises(ValueError, match="rowid 冲突"):
        index.insert([{"chunk_id": b, "text": OTHER}])


def test_冲突失败后库里没留半条(index):
    index.insert([{"chunk_id": cid(1), "text": DOC}])
    with pytest.raises(ValueError):
        index.insert([
            {"chunk_id": cid(2), "text": OTHER},
            {"chunk_id": cid(1), "text": OTHER},  # 第二条撞
        ])
    # 第一条也必须回滚
    assert index.search("词频", k=5) == []


def test_插入时不认识的键报错(index):
    with pytest.raises(ValueError, match="不认识的键"):
        index.insert([{"chunk_id": cid(1), "text": DOC, "向量": []}])


# ---------------------------------------------------------------- 注入


def test_tokenizer_是注入的(database):
    """换一个分词器，写入和查询都走它 —— 证明可插拔真的生效。"""

    class CharTokenizer:
        name = "char"

        def tokenize(self, text: str) -> list[str]:
            return [c for c in text if not c.isspace()]

    index = FTS5BM25Index(database.connection, CharTokenizer())
    index.open()
    index.insert([{"chunk_id": cid(1), "text": "甲乙丙"}])
    # 单字一定能查到 —— jieba 会把「甲乙丙」当一个词，char 版本必然切单字
    assert index.search("乙", k=5)[0].chunk_id == cid(1)


def test_没有_close_和_update(database, tok):
    """连接归调用方；FTS5 不支持原地改列 —— 假方法会让人以为收尾/原子了。"""
    index = FTS5BM25Index(database.connection, tok)
    assert not hasattr(index, "close")
    assert not hasattr(index, "update")


# ---------------------------------------------------------------- 与 vec 共存


def _both(database, tok):
    conn = database.connection
    vec = VecDB(conn)
    bm25 = FTS5BM25Index(conn, tok)
    vec.create_table(Schema(dim=4, metric="cosine", fields=("text",)))
    bm25.open()
    return vec, bm25


def test_与_VecDB_同连接共存(database, tok):
    vec, bm25 = _both(database, tok)
    vec.insert([{"chunk_id": cid(1), "vector": [1, 0, 0, 0], "text": DOC}])
    bm25.insert([{"chunk_id": cid(1), "text": DOC}])

    assert vec.get(cid(1)) is not None
    assert bm25.search("检索", k=5)[0].chunk_id == cid(1)


def test_跨表原子性(database, tok):
    """**B 方案的前提。** 同一个连接里，两边的写入被外层事务一起回滚。

    旧写法（`BEGIN`）在这里会抛 `cannot start a transaction within a transaction`。
    """
    conn = database.connection
    vec, bm25 = _both(database, tok)

    conn.execute("savepoint import")
    try:
        vec.insert([{"chunk_id": cid(1), "vector": [1, 0, 0, 0], "text": DOC}])
        bm25.insert([{"chunk_id": cid(1), "text": DOC}])
        conn.execute("rollback to import")
    finally:
        conn.execute("release import")

    assert vec.get(cid(1)) is None
    assert bm25.search("检索", k=5) == []


def test_跨表提交后两边都在(database, tok):
    conn = database.connection
    vec, bm25 = _both(database, tok)

    conn.execute("savepoint import")
    vec.insert([{"chunk_id": cid(1), "vector": [1, 0, 0, 0], "text": DOC}])
    bm25.insert([{"chunk_id": cid(1), "text": DOC}])
    conn.execute("release import")

    assert vec.get(cid(1)) is not None
    assert bm25.search("检索", k=5)[0].chunk_id == cid(1)


# ---------------------------------------------------------------- 重开库


def test_重开库数据还在(database, tok, tmp_path):
    index = FTS5BM25Index(database.connection, tok)
    index.open()
    index.insert([{"chunk_id": cid(1), "text": DOC}])
    database.close()

    again_db = Database(tmp_path / "bm25.db")
    try:
        again = FTS5BM25Index(again_db.connection, tok)
        again.open()
        assert again.search("检索", k=5)[0].chunk_id == cid(1)
    finally:
        again_db.close()
