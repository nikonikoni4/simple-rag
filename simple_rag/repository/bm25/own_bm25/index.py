"""`OwnBM25Index` —— 第三类 BM25：走调用方注入的 SQLite，表由本项目自己设计。

实例经 `create_bm25("own_bm25", ...)` 获得，本类**不在包出口里**。

它同时补上两个既有实现的缺口（见 [ADR 2026-10-02](../../../../docs/ADR/2026-10-02-新增第三类BM25实现.md)）：

- 相对 `fts5` —— `k1` / `b` **可调**，而且**换参数不用重建索引**：
  参数只进打分公式，不参与落盘，指向同一个库重新造实例即可。
- 相对 `rank_bm25` —— **语料不常驻 Python**。`rank_bm25` 把全库构造成
  `BM25Okapi` 的常驻结构（50,000 篇实测 1,927 MB，随 N 线性）；这里 Python
  只留当前这一篇的 token，统计量（`N` / `total_dl` / 每词 `df`）都在 SQLite 里。

**增量能力与 `fts5` 对齐**：`insert` / `delete` / `replace` 各自一个 `SAVEPOINT`
内完成，失败整批回滚。

分数**越小越相关**，且本实现**恒为负**：idf 用正 IDF（`log1p((N-df+0.5)/(df+0.5))`，
`df ≤ N` 时恒为正），不像 `rank_bm25` 那样有 epsilon floor、也不像 FTS5 那样把
负 idf 钳到 1e-6。统一接口没有承诺不同实现的分数逐位相同。
"""

from __future__ import annotations

import math
import sqlite3
from collections import Counter
from collections.abc import Iterable, Mapping
from contextlib import contextmanager

from simple_rag.tokenization import Tokenizer

from ..base import BM25Index, BM25SearchResult, check_k, check_record
from . import _store
from ._schema import ensure_schema, ensure_temp_tables

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


def check_params(k1: float, b: float) -> tuple[float, float]:
    """校验 `(k1, b)`，返回 `(float, float)`。

    **必须是有限值。** `NaN` / `Inf` 会让打分表达式静默产出 `nan` / `inf`
    （排序退化到无意义但**不报错**），`rank_bm25` 的既有校验漏掉了这层
    （`nan < 0` 是 False，所以 NaN 能溜过去），本实现显式拦。

    `k1 = 0` 合法 —— 此时单词贡献退化为 IDF，`tf` 不再影响该词得分，与
    `rank_bm25` 的参数域一致。
    """
    numbers: dict[str, float] = {}
    for name, value in (("k1", k1), ("b", b)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} 必须是数值，收到 {value!r}")
        try:
            number = float(value)
        except OverflowError as exc:  # 超大 int，如 10**400
            raise ValueError(f"{name} 超出 float 范围，收到 {value!r}") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} 必须是有限值，收到 {value!r}")
        numbers[name] = number
    if numbers["k1"] < 0:
        raise ValueError(f"k1 必须 >= 0，收到 {k1!r}")
    if not 0 <= numbers["b"] <= 1:
        raise ValueError(f"b 必须在 0~1 之间，收到 {b!r}")
    return numbers["k1"], numbers["b"]


@contextmanager
def _savepoint(conn: sqlite3.Connection, name: str):
    """把一段操作圈成一个保存点；异常时 `rollback to` + `release`。

    用 `SAVEPOINT` 而不是 `BEGIN` —— 调用方可能已经开着事务（要把本表和
    vec0 / FTS5 表的写入圈进同一个事务），那时 `BEGIN` 会报
    `cannot start a transaction within a transaction`。`SAVEPOINT` 在事务外
    自己开一个，在事务内成为子事务，**两种用法都对**。

    失败路径必须 `rollback to` **和** `release` 成对：只回滚不释放的话，
    保存点还留在栈上，外层事务一直挂着不提交，**而且不报错**。

    `release` 只释放本层保存点，不提交调用方的外层事务。
    """
    conn.execute(f"savepoint {name}")
    try:
        yield
    except BaseException:
        conn.execute(f"rollback to {name}")
        conn.execute(f"release {name}")
        raise
    conn.execute(f"release {name}")


class OwnBM25Index(BM25Index):
    """自设计 SQLite 倒排索引的 BM25。

    **不持有连接** —— 由调用方注入，本类不创建、不关闭，**没有 `close()`**。
    也不持有分词器之外的任何语料状态：`k1` / `b` 只是两个浮点数。

    连接是同步单线程用法：一次查询期间独占 TEMP 工作表，不支持同一连接上的
    重入查询（与既有实现一致）。
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        tokenizer: Tokenizer,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        """初始化。

        Args:
            conn: 调用方注入的连接，本类不持有其生命周期。
            tokenizer: 写入与查询共用的分词器。**换分词器要全库重建。**
            k1: 词频饱和参数，`>= 0` 且有限，默认 1.5。只影响打分，
                **换它不用重建索引**（指向同一个库新建实例即可）。
            b: 文档长度归一化强度，`0~1` 且有限，默认 0.75。同 `k1`。
        """
        self._conn = conn
        self._tokenizer = tokenizer
        self._k1, self._b = check_params(k1, b)
        # 稳定打分式要的两个预算值：字面写法在 k1 极大时会溢出（见 _store）。
        self._k1_plus_one = self._k1 + 1.0
        self._k1_ratio = self._k1 / self._k1_plus_one

    # ---------------------------------------------------------------- 打开

    def open(self) -> None:
        """幂等地把索引恢复到可查询：首建三张表，之后校验；并建 TEMP 工作表。

        **只建表与校验结构，不扫描语料、不恢复任何统计量** —— `N` / `total_dl`
        由写入路径原子维护，`open` 不重算它们。
        """
        with _savepoint(self._conn, "own_bm25_open"):
            ensure_schema(self._conn)
            ensure_temp_tables(self._conn)

    # ---------------------------------------------------------------- 写

    def insert(self, records: Iterable[Mapping]) -> int:
        """整批插入，返回写入篇数。

        `records` 每项 `{"chunk_id": str, "text": str}`，`text` 是**原文**。
        **逐篇流式处理** —— 不把整批原文物化进内存（`prepare_rows` 在这里不能用）。

        `chunk_id` 已存在时抛 `sqlite3.IntegrityError`（`docs.chunk_id` 的唯一
        约束）；批内重复抛 `ValueError`。两种都整批回滚。
        """
        with _savepoint(self._conn, "own_bm25_insert"):
            count, dl_total = self._stream_insert(self._conn, records)
            if count:
                _store.bump_stats(self._conn, count, dl_total)
            return count

    def delete(self, chunk_ids: Iterable[str]) -> int:
        """按 `chunk_id` 批量删，返回删除篇数。

        不存在的 id 不计数；同一批里重复给的 id 只算首次。
        **不把 `chunk_ids` 全量 list 化** —— 逐条按 `docs.chunk_id` 唯一索引定位，
        顺带避开 `IN (...)` 的参数个数上限。
        """
        conn = self._conn
        with _savepoint(conn, "own_bm25_delete"):
            _store.clear_seen(conn)
            deleted = 0
            dl_freed = 0
            for chunk_id in chunk_ids:
                if not isinstance(chunk_id, str):
                    raise TypeError(
                        f"chunk_id 必须是 str，收到 {type(chunk_id).__name__}"
                    )
                if not _store.mark_seen(conn, chunk_id):
                    continue
                found = _store.find_doc(conn, chunk_id)
                if found is None:
                    continue
                doc_id, dl = found
                _store.delete_doc(conn, doc_id)
                deleted += 1
                dl_freed += dl
            if deleted:
                _store.bump_stats(conn, -deleted, -dl_freed)
            return deleted

    def replace(self, records: Iterable[Mapping]) -> int:
        """覆盖或插入（upsert），返回输入篇数。

        已存在的 `chunk_id` 先按 `delete` 的方式移除旧数据再写新数据 ——
        **新数据即使是空文本也覆盖旧的词频**（否则「改短」会留下幽灵命中）。
        """
        conn = self._conn
        with _savepoint(conn, "own_bm25_replace"):
            _store.clear_seen(conn)
            count = 0
            n_delta = 0
            dl_delta = 0
            for record in records:
                chunk_id, text = check_record(record)
                if not _store.mark_seen(conn, chunk_id):
                    raise ValueError(f"批内 chunk_id 重复：{chunk_id!r}")
                found = _store.find_doc(conn, chunk_id)
                if found is not None:
                    _store.delete_doc(conn, found[0])
                    n_delta -= 1
                    dl_delta -= found[1]
                dl_delta += _store.insert_doc(
                    conn, chunk_id, self._tokenizer.tokenize(text)
                )
                n_delta += 1
                count += 1
            if count:
                _store.bump_stats(conn, n_delta, dl_delta)
            return count

    def rebuild(self, records: Iterable[Mapping]) -> int:
        """全量重建，返回写入篇数；原有数据**全部丢弃**，空批也清空。

        与 `fts5` 的 rebuild 语义一致。实现在一个保存点内清空 + 流式重填，
        **不需要删表，也不把全库留在内存**；中途失败回到旧索引。
        """
        conn = self._conn
        with _savepoint(conn, "own_bm25_rebuild"):
            _store.clear_index(conn)
            count, dl_total = self._stream_insert(conn, records)
            if count:
                _store.bump_stats(conn, count, dl_total)
            return count

    def _stream_insert(
        self, conn: sqlite3.Connection, records: Iterable[Mapping]
    ) -> tuple[int, int]:
        """逐篇 `check_record → 分词 → 写库`，返回 `(篇数, 总长度)`。

        **每篇的 token / Counter 用完即弃**，只有当前这一篇在 Python 内存里；
        整批原文只在调用方的 iterable 里各出现一次。

        批内重复用连接局部 TEMP 表判，不建与批大小同步增长的 Python `set`。
        """
        _store.clear_seen(conn)
        count = 0
        dl_total = 0
        for record in records:
            chunk_id, text = check_record(record)
            if not _store.mark_seen(conn, chunk_id):
                raise ValueError(f"批内 chunk_id 重复：{chunk_id!r}")
            dl_total += _store.insert_doc(
                conn, chunk_id, self._tokenizer.tokenize(text)
            )
            count += 1
        return count, dl_total

    # ---------------------------------------------------------------- 读

    def search(self, query: str, k: int) -> list[BM25SearchResult]:
        """关键词查询，按相关度升序（最相关在前）。

        `query` 是**原文**，分词在本类内做；分不出 token 时返回 `[]`。
        重复的查询词按出现次数加权（`qtf`），多个词之间是 `OR`。

        读流程整个包在一个保存点里，让 `N` / `total_dl` / 每词 `df` / 候选
        属于**同一个 SQLite 快照** —— 否则半路来了写入（另一条连接）会让
        分子分母来自不同版本。

        打分与候选聚合在 SQLite 里完成，Python 只算查询词的 IDF，只拿最终
        `k` 行。
        """
        k = check_k(k)
        if k == 0:
            return []
        tokens = self._tokenizer.tokenize(query)
        if not tokens:
            return []
        conn = self._conn
        with _savepoint(conn, "own_bm25_search"):
            n, total_dl = _store.read_stats(conn)
            # N = 0 或全库长度为 0 时 avgdl 无定义 —— 也说明没有任何 posting
            # 可命中，直接空结果（顺带避开除零）。
            if n == 0 or total_dl == 0:
                return []
            avgdl = total_dl / n
            _store.clear_query(conn)
            _store.fill_query(
                conn, self._idf_rows(conn, Counter(tokens), n)
            )
            rows = _store.run_search(
                conn,
                k1_plus_one=self._k1_plus_one,
                k1_ratio=self._k1_ratio,
                b=self._b,
                avgdl=avgdl,
                k=k,
            )
        return [BM25SearchResult(chunk_id=row[0], score=row[1]) for row in rows]

    @staticmethod
    def _idf_rows(
        conn: sqlite3.Connection, qtf: Counter, n: int
    ) -> Iterable[tuple[str, float, int]]:
        """本次查询每个不同词一行 `(term, idf, qtf)`；`df = 0` 的词跳过。

        IDF 用正 IDF `log1p((N - df + 0.5) / (df + 0.5))`：`df <= N` 时恒为正，
        没有 epsilon floor 也没有下限钳制 —— 高频词仍参与打分。
        """
        for term, count in qtf.items():
            df = _store.count_df(conn, term)
            if df == 0:
                continue
            yield term, math.log1p((n - df + 0.5) / (df + 0.5)), count
