"""`BM25Index` —— 统一接口的抽象基类。

调用方只看得到本类（经 `create_bm25` 工厂拿到实例），不感知具体实现。
三个实现各自内聚在 `fts5_bm25/`、`okapi_bm25/` 与 `own_bm25/` 子包里。

**接口语义（所有实现必须一致的部分）：**

- `open()` 幂等：把索引恢复到「可查询」。首次是建，之后是校验/加载。
- 写入都吃 `records`：每项 `{"chunk_id": str, "text": str}`，`text` 是**原文**，
  分词在实现内部做。
- `search` 返回的 `score` 统一是**负数、越小越相关**（见 `types.BM25SearchResult`）。
- **不支持的操作必须抛 `NotImplementedError`**，不许静默 no-op —— 调用方换实现
  时，能力差异要在第一时间暴露（如 rank_bm25 的 `insert` / `delete` / `replace`）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping

from .types import BM25SearchResult


class BM25Index(ABC):
    """BM25 索引的统一接口。实例经 `create_bm25(impl=...)` 获得。"""

    @abstractmethod
    def open(self) -> None:
        """幂等地把索引恢复到可查询状态。

        - fts5：表不存在则建（首次），存在则校验 DDL 一致；
        - rank_bm25：构造时给了 `persist_path` 且文件在则加载，否则空索引；
        - own_bm25：三张表不存在则建，存在则校验结构与索引；并建 TEMP 工作表。
        """

    @abstractmethod
    def rebuild(self, records: Iterable[Mapping]) -> int:
        """全量重建，返回写入条数。原有数据**全部丢弃**。

        - fts5：删表 + 重建 + 全量插入（一个事务内）；
        - rank_bm25：重新分词、重算 `BM25Okapi`，有 `persist_path` 则写盘；
        - own_bm25：一个保存点内清空三张表 + 流式重填（不删表）。

        本方法是 rank_bm25 唯一的写入口 —— 它没有增量能力。
        """

    @abstractmethod
    def insert(self, records: Iterable[Mapping]) -> int:
        """增量插入，返回写入条数。`chunk_id` 已存在时由实现报错。"""

    @abstractmethod
    def delete(self, chunk_ids: Iterable[str]) -> int:
        """按 `chunk_id` 批量删，返回删除行数。"""

    @abstractmethod
    def replace(self, records: Iterable[Mapping]) -> int:
        """覆盖或插入（upsert），返回写入条数。

        `chunk_id` 已存在则删旧插新（一个事务内），不存在也直接写入 ——
        调用方不用先判断存在性。
        """

    @abstractmethod
    def search(self, query: str, k: int) -> list[BM25SearchResult]:
        """关键词查询，按相关度排序（最相关在前）。

        `query` 是**原文**，分词在实现内部做；分不出 token 时（空串、纯空白、
        纯标点）返回 `[]`。只返回有命中的文档 —— 查询词一个都不在文中的
        文档不进结果（即使它 score 排得进前 k）。
        """
