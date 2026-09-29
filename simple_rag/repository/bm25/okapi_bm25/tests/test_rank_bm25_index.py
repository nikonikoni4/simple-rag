"""`RankBM25Index` 的测试 —— 不碰 SQLite，只测内存索引与 pickle 往返。"""

from __future__ import annotations

import pickle

import pytest

from simple_rag.repository.bm25 import BM25SearchResult
from simple_rag.repository.bm25.okapi_bm25 import RankBM25Index
from simple_rag.repository.bm25.okapi_bm25.persistence import load, save
from simple_rag.tokenization import TokenizerFactory

DOC = "向量检索是基于语义的检索方法"
OTHER = "BM25 是基于词频的检索方法"


def cid(n: int) -> str:
    """任意 str 都行 —— rank_bm25 实现的 chunk_id 只是普通 key，不要求 hex。"""
    return f"chunk-{n}"


@pytest.fixture
def tok():
    return TokenizerFactory.create("jieba")


@pytest.fixture
def index(tok):
    return RankBM25Index(tok)


# ---------------------------------------------------------------- 基本流程


def test_rebuild后能查到(index):
    assert index.rebuild([{"chunk_id": cid(1), "text": DOC}]) == 1
    hits = index.search("检索", k=5)
    assert [h.chunk_id for h in hits] == [cid(1)]
    assert isinstance(hits[0], BM25SearchResult)


def test_常规语料分数为负且升序(index):
    """查询词只出现在少数文档时 idf 为正 —— 取负后是负数（常规情形）。"""
    index.rebuild([
        {"chunk_id": cid(1), "text": "检索 语义 甲"},
        {"chunk_id": cid(2), "text": "无关 乙"},
        {"chunk_id": cid(3), "text": "别的 丙"},
    ])
    hits = index.search("检索", k=5)
    assert [h.chunk_id for h in hits] == [cid(1)]
    assert hits[0].score < 0


def test_退化语料分数可能为正(index):
    """单文档语料：所有词的 idf 都为负，被 epsilon floor 成负数 —— 取负后为正。

    **方向契约（越小越相关）不受影响**；命中文档按词判断，不被符号误杀。
    这是 `BM25Okapi` 的退化行为，不是 bug，写成断言钉住。
    """
    index.rebuild([{"chunk_id": cid(1), "text": "检索 语义"}])
    hits = index.search("检索", k=5)
    assert [h.chunk_id for h in hits] == [cid(1)]
    assert hits[0].score > 0


def test_不相关的查不到(index):
    index.rebuild([{"chunk_id": cid(1), "text": DOC}])
    assert index.search("量子力学", k=5) == []


def test_未命中文档被过滤(index):
    """`get_scores` 给全量文档打分 —— 查询词一个都不在文中的不能进结果。

    这里 cid(1) 的 idf 恰好是 0（2 篇文档里 1 篇含「检索」），它**必须**
    留在结果里 —— 证明过滤按词命中，不按分数符号。
    """
    index.rebuild([
        {"chunk_id": cid(1), "text": DOC},
        {"chunk_id": cid(2), "text": "完全无关的内容"},
    ])
    hits = index.search("检索", k=10)  # k 故意给大
    assert [h.chunk_id for h in hits] == [cid(1)]


def test_k_为0返回空(index):
    index.rebuild([{"chunk_id": cid(1), "text": DOC}])
    assert index.search("检索", k=0) == []


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_空查询返回空(index, blank):
    index.rebuild([{"chunk_id": cid(1), "text": DOC}])
    assert index.search(blank, k=5) == []


# ---------------------------------------------------------------- 空索引


def test_没rebuild过search返回空(index):
    """空索引（okapi 为 None）是合法可查询状态。"""
    assert index.search("检索", k=5) == []


def test_rebuild空批清空(index):
    index.rebuild([{"chunk_id": cid(1), "text": DOC}])
    assert index.rebuild([]) == 0
    assert index.search("检索", k=5) == []


def test_全空text语料不炸(index):
    """语料有条目但全是空 token（text 为空串 / 纯空白）—— 库内
    `idf_sum / len(idf)` 是另一个 0/0，必须归为空索引而不是崩。

    条目照常保留（返回条数对），空文档永不命中 —— 和 fts5 的
    「空文本入库、永不命中」对齐。
    """
    assert index.rebuild([
        {"chunk_id": cid(1), "text": ""},
        {"chunk_id": cid(2), "text": "   "},
    ]) == 2
    assert index.search("检索", k=5) == []


def test_混合语料里空text文档永不命中(index):
    index.rebuild([
        {"chunk_id": cid(1), "text": ""},
        {"chunk_id": cid(2), "text": DOC},
    ])
    hits = index.search("检索", k=5)
    assert [h.chunk_id for h in hits] == [cid(2)]


# ---------------------------------------------------------------- 没有增量


@pytest.mark.parametrize(
    "method, args",
    [("insert", ([{"chunk_id": cid(1), "text": DOC}],)),
     ("delete", ([cid(1)],)),
     ("replace", ([{"chunk_id": cid(1), "text": DOC}],))],
)
def test_增量操作抛_NotImplementedError(index, method, args):
    with pytest.raises(NotImplementedError, match="rebuild"):
        getattr(index, method)(*args)


# ---------------------------------------------------------------- 调参


def test_k1_b_可调且影响分数(tok):
    """这一路存在的理由 —— FTS5 调不了 k1/b，这里必须真生效。

    三篇文档长度递增（2/3/4 token）且都含查询词 —— b 的长度归一化和
    k1 的词频饱和在两套参数下必然给出不同的分数组合。
    """
    records = [
        {"chunk_id": cid(0), "text": "检索 甲"},
        {"chunk_id": cid(1), "text": "检索 甲 乙"},
        {"chunk_id": cid(2), "text": "检索 甲 乙 丙"},
    ]
    default = RankBM25Index(tok)
    default.rebuild(records)
    tuned = RankBM25Index(tok, k1=1.2, b=0.3)
    tuned.rebuild(records)
    assert default.search("检索", k=3) != tuned.search("检索", k=3)


def test_默认k1是1_5(tok):
    index = RankBM25Index(tok)
    assert index._k1 == 1.5
    assert index._b == 0.75


@pytest.mark.parametrize("bad_k1", [-1, "1.5", True, None])
def test_非法k1报错(tok, bad_k1):
    with pytest.raises(ValueError, match="k1"):
        RankBM25Index(tok, k1=bad_k1)


@pytest.mark.parametrize("bad_b", [-0.1, 1.1, "0.5", None])
def test_非法b报错(tok, bad_b):
    with pytest.raises(ValueError, match="b"):
        RankBM25Index(tok, b=bad_b)


# ---------------------------------------------------------------- pickle 持久化


def test_persist_path_文件不存在_open后空索引(tok, tmp_path):
    index = RankBM25Index(tok, persist_path=tmp_path / "bm25.pkl")
    index.open()
    assert index.search("检索", k=5) == []


def test_rebuild自动写盘_新实例open恢复(tok, tmp_path):
    path = tmp_path / "bm25.pkl"
    first = RankBM25Index(tok, persist_path=path)
    first.rebuild([{"chunk_id": cid(1), "text": DOC}])
    assert path.exists()

    again = RankBM25Index(tok, persist_path=path)
    again.open()
    assert [h.chunk_id for h in again.search("检索", k=5)] == [cid(1)]


def test_open幂等(tok, tmp_path):
    path = tmp_path / "bm25.pkl"
    index = RankBM25Index(tok, persist_path=path)
    index.rebuild([{"chunk_id": cid(1), "text": DOC}])
    index.open()
    index.open()
    assert len(index.search("检索", k=5)) == 1


def test_rebuild空批也写盘(tok, tmp_path):
    path = tmp_path / "bm25.pkl"
    index = RankBM25Index(tok, persist_path=path)
    index.rebuild([{"chunk_id": cid(1), "text": DOC}])
    index.rebuild([])
    again = RankBM25Index(tok, persist_path=path)
    again.open()
    assert again.search("检索", k=5) == []


def test_rebuild写盘失败_内存保持旧数据(tok, tmp_path, monkeypatch):
    """先写盘再换内存 —— 写盘炸（磁盘满 / 权限）时活索引不能已经换成新数据，
    否则调用方拿到异常却不知道内存里是什么。"""
    from simple_rag.repository.bm25.okapi_bm25 import persistence

    index = RankBM25Index(tok, persist_path=tmp_path / "bm25.pkl")
    index.rebuild([{"chunk_id": cid(1), "text": DOC}])

    def boom(path, records):
        raise OSError("磁盘满了（测试模拟）")

    monkeypatch.setattr(persistence, "save", boom)
    with pytest.raises(OSError, match="磁盘满"):
        index.rebuild([{"chunk_id": cid(2), "text": OTHER}])
    # 内存还是旧数据
    assert [h.chunk_id for h in index.search("检索", k=5)] == [cid(1)]
    assert index.search("词频", k=5) == []


def test_坏pickle文件open报错(tok, tmp_path):
    path = tmp_path / "bm25.pkl"
    path.write_bytes(b"not a pickle")
    index = RankBM25Index(tok, persist_path=path)
    with pytest.raises(ValueError, match="读不出来"):
        index.open()


def test_格式版本不认的文件报错(tok, tmp_path):
    path = tmp_path / "bm25.pkl"
    path.write_bytes(pickle.dumps({"version": 999, "records": []}))
    index = RankBM25Index(tok, persist_path=path)
    with pytest.raises(ValueError, match="版本"):
        index.open()


# ---------------------------------------------------------------- persistence 纯函数


def test_save_load_往返(tmp_path):
    path = tmp_path / "bm25.pkl"
    data = [("a", ["检索", "语义"]), ("b", ["词频"])]
    save(path, data)
    assert load(path) == data


def test_save_不生成残留临时文件(tmp_path):
    path = tmp_path / "bm25.pkl"
    save(path, [])
    assert [p.name for p in tmp_path.iterdir()] == ["bm25.pkl"]
