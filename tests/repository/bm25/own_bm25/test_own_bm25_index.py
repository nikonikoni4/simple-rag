"""own_bm25 的评分与查询 —— 每条都对手算值或独立 oracle 核对。

打分是这一路实现存在的理由（`fts5` 调不了参），所以这里不满足于「能查回」：
`k1` / `b` 的每个边界都要和 oracle 对上，极端 `k1` 还要额外验证分数**有限**。
"""

from __future__ import annotations

import math
import sqlite3
from collections import Counter

import pytest

from simple_rag.repository.bm25 import create_bm25
from simple_rag.repository.bm25.base.validation import K_MAX

# 打分验收常用的小语料：含重复词、长文档、短文档、只共享部分词、空文档
DOCS = [
    ("d1", "a a b".split()),
    ("d2", "a".split()),
    ("d3", "b c c c".split()),
    ("d4", "a b c d d d d d".split()),
    ("d5", []),
]

# ---------------------------------------------------------------- oracle


def _idf(n: int, df: int) -> float:
    return math.log1p((n - df + 0.5) / (df + 0.5))


def stable_oracle(docs, query, k1, b):
    """正 IDF 的 BM25，用**稳定代数式**（与生产 SQL 同式，但独立实现）。

    稳定式是唯一在 `k1 = 1e308` 下也算得出来的写法，所以极端参数的验收
    只能对它做。
    """
    n = len(docs)
    if n == 0:
        return []
    total_dl = sum(len(tokens) for _, tokens in docs)
    avgdl = total_dl / n
    qtf = Counter(query)
    df = {term: sum(1 for _, tokens in docs if term in tokens) for term in qtf}
    k1_plus_one = k1 + 1.0
    k1_ratio = k1 / k1_plus_one

    scored = []
    for chunk_id, tokens in docs:
        tf = Counter(tokens)
        dl = len(tokens)
        total = 0.0
        for term, q in qtf.items():
            if df[term] == 0 or tf.get(term, 0) == 0:
                continue
            contribution = tf[term] / (
                tf[term] / k1_plus_one + k1_ratio * (1.0 - b + b * dl / avgdl)
            )
            total += q * _idf(n, df[term]) * contribution
        if total:
            scored.append((chunk_id, -total))
    scored.sort(key=lambda pair: (pair[1], pair[0]))  # 分数升序，同分按 chunk_id
    return scored


def naive_oracle(docs, query, k1, b):
    """**字面公式** `idf * tf*(k1+1) / (tf + k1*(1-b+b*dl/avgdl))`。

    只在 `k1` 不极端时可用（`tf*(k1+1)` 会溢出）。它与 `stable_oracle` 代数等价 ——
    两者对得上，才说明生产 SQL 的改写没改语义。
    """
    n = len(docs)
    if n == 0:
        return []
    total_dl = sum(len(tokens) for _, tokens in docs)
    avgdl = total_dl / n
    qtf = Counter(query)
    df = {term: sum(1 for _, tokens in docs if term in tokens) for term in qtf}

    scored = []
    for chunk_id, tokens in docs:
        tf = Counter(tokens)
        dl = len(tokens)
        total = 0.0
        for term, q in qtf.items():
            if df[term] == 0 or tf.get(term, 0) == 0:
                continue
            contribution = (
                _idf(n, df[term])
                * tf[term]
                * (k1 + 1)
                / (tf[term] + k1 * (1 - b + b * dl / avgdl))
            )
            total += q * contribution
        if total:
            scored.append((chunk_id, -total))
    scored.sort(key=lambda pair: (pair[1], pair[0]))
    return scored


def make(conn, tok, k1, b):
    """同一个库、换一组参数再造一个实例 —— 参数只进打分，不进落盘。"""
    instance = create_bm25("own_bm25", tokenizer=tok, conn=conn, k1=k1, b=b)
    instance.open()
    return instance


def load(conn, tok, docs):
    """把 `(chunk_id, [tokens])` 灌进库里，返回默认参数的索引实例。"""
    instance = make(conn, tok, 1.5, 0.75)
    instance.rebuild([{"chunk_id": cid, "text": " ".join(t)} for cid, t in docs])
    return instance


# ---------------------------------------------------------------- 手算样本


def test_手算小语料_重复查询词加权(conn, tok):
    """`a a b` / `a` / 空文档，查询 `a a`。

    手算（k1=1.5, b=0.75）：N=3、total_dl=4、avgdl=4/3、df(a)=2
    - idf(a) = log1p((3-2+0.5)/(2+0.5)) = log1p(0.6)
    - doc `a`：tf=2, dl=3 -> 相对长度 3/(4/3) = 2.25
    - doc `b`：tf=1, dl=1 -> 相对长度 0.75，**更短所以得分更高**
    - doc `c` 没有 a，不进结果
    """
    index = load(
        conn,
        tok,
        [("a", "a a b".split()), ("b", "a".split()), ("c", [])],
    )
    hits = index.search("a a", k=10)
    assert [h.chunk_id for h in hits] == ["b", "a"]

    # 手算：字面公式 idf * tf*(k1+1) / (tf + k1*(1-b+b*dl/avgdl))
    idf = math.log1p((3 - 2 + 0.5) / (2 + 0.5))
    avgdl = 4 / 3

    def term_score(tf: int, dl: int) -> float:
        return idf * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avgdl))

    assert hits[0].score == pytest.approx(-2 * term_score(tf=1, dl=1))
    assert hits[1].score == pytest.approx(-2 * term_score(tf=2, dl=3))

    # 统计量：N 计入空文档，total_dl 只加真实长度
    assert conn.execute("select n, total_dl from own_bm25_stats").fetchone() == (3, 4)
    assert conn.execute(
        "select count(*) from own_bm25_postings where term = 'a'"
    ).fetchone() == (2,)
    assert conn.execute(
        "select count(*) from own_bm25_postings where term = 'b'"
    ).fetchone() == (1,)


def test_重复查询词按次数加权(conn, tok):
    """`a a` 的每个词得分是不带重复时的两倍 —— qtf 生效。"""
    index = load(conn, tok, [("x", "a b".split())])
    once = index.search("a", k=5)[0].score
    twice = index.search("a a", k=5)[0].score
    assert twice == pytest.approx(2 * once)


# ---------------------------------------------------------------- 对 oracle


@pytest.mark.parametrize("k1", [0.0, 1e-300, 0.3, 1.5, 2.0, 1e308])
@pytest.mark.parametrize("b", [0.0, 0.75, 1.0])
@pytest.mark.parametrize("query", ["a", "a a", "b c", "a b c", "zzz", "a zzz"])
def test_与稳定oracle一致(conn, tok, k1, b, query):
    load(conn, tok, DOCS)
    index = make(conn, tok, k1, b)
    expected = stable_oracle(DOCS, query.split(), k1, b)
    hits = index.search(query, k=100)
    assert [h.chunk_id for h in hits] == [cid for cid, _ in expected]
    for hit, (_, score) in zip(hits, expected):
        assert hit.score == pytest.approx(score, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("k1", [0.0, 0.3, 1.5, 2.0])
@pytest.mark.parametrize("b", [0.0, 0.75, 1.0])
def test_稳定式与字面公式等价(conn, tok, k1, b):
    """生产 SQL 的代数改写必须与**字面公式**同值 —— 否则就是偷偷换了打分。

    字面式在 k1 极大时溢出，所以这组只在常规参数上跑。
    """
    load(conn, tok, DOCS)
    index = make(conn, tok, k1, b)
    hits = index.search("a b c", k=100)
    expected = naive_oracle(DOCS, "a b c".split(), k1, b)
    assert [h.chunk_id for h in hits] == [cid for cid, _ in expected]
    for hit, (_, score) in zip(hits, expected):
        assert hit.score == pytest.approx(score, rel=1e-12, abs=1e-15)


def test_k1为0时单词贡献退化为IDF(conn, tok):
    """k1=0 合法：tf 不再影响该词得分，长文档与短文档同分。"""
    load(conn, tok, [("long", "a a a a a".split()), ("short", "a".split())])
    index = make(conn, tok, 0.0, 0.75)
    hits = index.search("a", k=10)
    idf = math.log1p((2 - 2 + 0.5) / (2 + 0.5))
    assert [h.chunk_id for h in hits] == ["long", "short"]  # 同分 -> chunk_id 升序
    assert hits[0].score == pytest.approx(-idf)
    assert hits[1].score == pytest.approx(-idf)


def test_极端k1得分有限且为负(conn, tok):
    """k1=1e308 时字面式会 `inf` —— 稳定式必须给出有限负分。"""
    docs = [("a", "x x x".split()), ("b", "x".split())]
    load(conn, tok, docs)
    index = make(conn, tok, 1e308, 0.0)
    hits = index.search("x", k=10)
    expected = stable_oracle(docs, ["x"], 1e308, 0.0)
    assert [h.chunk_id for h in hits] == [cid for cid, _ in expected]
    for hit in hits:
        assert math.isfinite(hit.score)
        assert hit.score < 0


# ---------------------------------------------------------------- 排序与过滤


def test_同分按chunk_id升序(conn, tok):
    index = load(conn, tok, [("b", ["x"]), ("a", ["x"]), ("c", ["x"])])
    assert [h.chunk_id for h in index.search("x", k=10)] == ["a", "b", "c"]


def test_只返回命中文档(conn, tok):
    index = load(conn, tok, [("hit", "检索 检索".split()), ("miss", "完全 无关".split())])
    assert [h.chunk_id for h in index.search("检索", k=10)] == ["hit"]


def test_k超过命中数只返回命中(conn, tok):
    index = load(conn, tok, [("a", ["x"]), ("b", ["y"])])
    assert len(index.search("x", k=K_MAX)) == 1


@pytest.mark.parametrize("query", ["", "   ", "\t\n", "zzz"])
def test_无有效查询词或全无命中返回空(conn, tok, query):
    index = load(conn, tok, [("a", "x y".split())])
    assert index.search(query, k=10) == []


def test_k为0返回空(conn, tok):
    index = load(conn, tok, [("a", ["x"])])
    assert index.search("x", k=0) == []


@pytest.mark.parametrize("bad", [4097, -1, True, "1", 1.0, None])
def test_非法k报错(conn, tok, bad):
    index = load(conn, tok, [("a", ["x"])])
    with pytest.raises((TypeError, ValueError)):
        index.search("x", k=bad)


def test_空索引返回空(conn, tok):
    index = make(conn, tok, 1.5, 0.75)
    assert index.search("x", k=10) == []


def test_token按精确匹配_不二次归一化(conn, tok):
    """token 是分词器的输出，SQLite 默认 BINARY —— 大小写与标点都不再动。"""
    index = load(conn, tok, [("a", "检索。 !! Abc".split())])
    assert [h.chunk_id for h in index.search("检索。", k=10)] == ["a"]
    assert index.search("abc", k=10) == []  # 大小写不同 -> 不命中
    assert [h.chunk_id for h in index.search("Abc", k=10)] == ["a"]


# ---------------------------------------------------------------- 查询词规模


@pytest.mark.skipif(
    not hasattr(sqlite3.Connection, "setlimit"),
    reason="需要 Python 3.11+ 的 Connection.setlimit 才能压低参数上限",
)
def test_查询词数远超参数上限(tmp_path, tok):
    """把连接的变量上限压到 8，再发一条 200 个不同词的查询。

    打分用的是 TEMP 词表连接，不是 `term in (?, ?, ...)` —— 拼 IN 的写法在这里
    会 `sqlite3.OperationalError: too many SQL variables`。**没有新增查询长度上限**：
    词再翻十倍也只是 TEMP 表行数变多。
    """
    from simple_rag.db import Database

    db = Database(tmp_path / "many_terms.db")
    try:
        db.connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 8)
        index = create_bm25("own_bm25", tokenizer=tok, conn=db.connection)
        index.open()
        index.rebuild([{"chunk_id": f"d{i:04d}", "text": f"t{i}"} for i in range(200)])
        query = " ".join(f"t{i}" for i in range(200))
        hits = index.search(query, k=10)
        assert len(hits) == 10
        assert all(math.isfinite(h.score) for h in hits)
    finally:
        db.close()
