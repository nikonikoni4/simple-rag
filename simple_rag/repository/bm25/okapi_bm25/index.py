"""`RankBM25Index` —— 基于 rank_bm25 库 `BM25Okapi` 的 BM25 实现。

实例经 `create_bm25("rank_bm25", ...)` 获得，本类**不在包出口里**。

与 fts5 实现的本质差异（都源自 `BM25Okapi` 的构造即终态）：

- **纯内存** —— 自带的持久化是构造时给 `persist_path`（pickle 纯数据，
  见 `persistence.py`）；不给就是纯内存，重启后调用方自己全量 `rebuild`。
- **没有增量** —— `insert` / `delete` / `replace` 直接抛 `NotImplementedError`。
  不做「写盘后全量重算」式的伪增量：一次插入一次 O(N) 重算，是性能陷阱。
- **`k1` / `b` 可调** —— 这一路存在的理由（FTS5 硬编码 1.2 / 0.75）。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from pathlib import Path

from rank_bm25 import BM25Okapi

from simple_rag.tokenization import Tokenizer

from ..base import BM25Index, BM25SearchResult, check_k, prepare_rows
from . import persistence

logger = logging.getLogger(__name__)

_UNSUPPORTED = (
    "rank_bm25 实现没有增量能力（BM25Okapi 构造即终态）："
    "改数据请攒齐全量后 rebuild"
)


class RankBM25Index(BM25Index):
    """rank_bm25 关键词索引。"""

    def __init__(
        self,
        tokenizer: Tokenizer,
        k1: float = 1.5,
        b: float = 0.75,
        persist_path: str | Path | None = None,
    ) -> None:
        """初始化。

        Args:
            tokenizer: 注入的分词器，写入和查询共用。
            k1: BM25 的词频饱和参数，>= 0。FTS5 硬编码 1.2，本实现默认取
                rank_bm25 库的 1.5；要和 FTS5 对齐就显式传 1.2。
            b: BM25 的文档长度归一化强度，0~1。
            persist_path: 给了则 rebuild 自动写盘、open 自动加载；
                None 表示纯内存，重启后由调用方全量 rebuild。
        """
        if not isinstance(k1, (int, float)) or isinstance(k1, bool) or k1 < 0:
            raise ValueError(f"k1 必须是非负数，收到 {k1!r}")
        if not isinstance(b, (int, float)) or isinstance(b, bool) or not 0 <= b <= 1:
            raise ValueError(f"b 必须在 0~1 之间，收到 {b!r}")

        self._tokenizer = tokenizer
        self._k1 = float(k1)
        self._b = float(b)
        self._persist_path = Path(persist_path) if persist_path is not None else None
        # 空语料构造 BM25Okapi 会 0/0 除零 —— 空索引用 None 表示，
        # search 对 None 返回 []。
        self._okapi: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        # 每篇文档的 token 集合 —— 「有没有命中」按词判断，不按分数符号
        # （见 search 的注释）。
        self._doc_tokens: list[frozenset[str]] = []

    # ---------------------------------------------------------------- 打开

    def open(self) -> None:
        """幂等地恢复：`persist_path` 文件在则加载，不在 / 没配则空索引。

        空索引的 search 返回 `[]` —— 对本实现，「没有持久化文件」和
        「持久化了空语料」是同一个可查询状态。
        """
        if self._persist_path is None or not self._persist_path.exists():
            return
        records = persistence.load(self._persist_path)
        self._build(records)

    # ---------------------------------------------------------------- 写

    def rebuild(self, records: Iterable[Mapping]) -> int:
        """全量重建，返回写入条数；配了 `persist_path` 则自动写盘。

        空批把索引清空（`BM25Okapi` 回到 None）—— 和 fts5 的「重建为空」
        语义一致。

        **先写盘再换内存**：写盘失败（磁盘满 / 权限）时内存索引保持旧数据，
        调用方拿到异常就知道活索引没变 —— 不会出现「内存已是新数据、
        磁盘还是旧的，下次 open 又静默回退」的撕裂。
        """
        rows = prepare_rows(records)
        data = [
            (chunk_id, self._tokenizer.tokenize(text)) for chunk_id, text in rows
        ]
        if self._persist_path is not None:
            persistence.save(self._persist_path, data)
        self._build(data)
        return len(data)

    def insert(self, records: Iterable[Mapping]) -> int:
        raise NotImplementedError(_UNSUPPORTED)

    def delete(self, chunk_ids: Iterable[str]) -> int:
        raise NotImplementedError(_UNSUPPORTED)

    def replace(self, records: Iterable[Mapping]) -> int:
        raise NotImplementedError(_UNSUPPORTED)

    # ---------------------------------------------------------------- 读

    def search(self, query: str, k: int) -> list[BM25SearchResult]:
        """关键词查询，按相关度排序（最相关在前）。

        `query` 是**原文**，分词在本类内做；分不出 token 时返回 `[]`。
        空索引（没 open 到数据 / rebuild 过空批）也返回 `[]`。

        **「有没有命中」按词判断，不按分数符号。** `BM25Okapi` 的 idf 在退化
        语料下（词出现在过半文档，极端如单文档语料）会被 epsilon floor 成
        负数 —— 命中文档照样得分负分。按分数过滤会把它们误杀，所以用
        每篇文档的 token 集合判断命中。

        分数对 `BM25Okapi` 的原始分**取负** —— 统一接口的契约是「越小越相关」，
        方向恒成立；**符号**不是契约（常规语料为负，退化语料可能为正）。
        """
        k = check_k(k)
        if k == 0 or self._okapi is None:
            return []
        tokens = self._tokenizer.tokenize(query)
        if not tokens:
            return []

        query_tokens = set(tokens)
        scores = self._okapi.get_scores(tokens)
        matched = [
            (float(scores[i]), i)
            for i, doc_tokens in enumerate(self._doc_tokens)
            if not doc_tokens.isdisjoint(query_tokens)
        ]
        matched.sort(key=lambda pair: -pair[0])
        return [
            BM25SearchResult(chunk_id=self._chunk_ids[i], score=-score)
            for score, i in matched[:k]
        ]

    # ---------------------------------------------------------------- 内部

    def _build(self, data: persistence.Records) -> None:
        """纯数据 → 内存索引，全部构造成功才提交状态。

        两个坑都实测过：

        - 空语料构造 `BM25Okapi` 会 0/0 除零；**语料有条目但全是空 token 列表**
          （text 是空串 / 纯空白）时，库内另一个 0/0（`idf_sum / len(idf)`）
          照样炸 —— 所以「无任何词条」也归为空索引（okapi 为 None），
          条目本身照常保留（和 fts5 的「空文本入库、永不命中」对齐）。
        - 先赋 `self._chunk_ids` 再构造 okapi，构造抛错时一半新一半旧 ——
          状态撕裂。局部变量构造完再一次性提交。
        """
        chunk_ids = [chunk_id for chunk_id, _ in data]
        doc_tokens = [frozenset(tokens) for _, tokens in data]
        corpus = [tokens for _, tokens in data]
        okapi = BM25Okapi(corpus, k1=self._k1, b=self._b) if any(corpus) else None
        self._chunk_ids = chunk_ids
        self._doc_tokens = doc_tokens
        self._okapi = okapi
