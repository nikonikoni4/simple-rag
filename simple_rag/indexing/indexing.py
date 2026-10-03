"""索引构建的编排层：把「建索引」这条链路的内部环节整合到一起。

对外两个名字：`RagIndexStrategy` 是配置，`RagIndexingPipeline` 是执行。

链路是四步，中间产物一路传下去 —— 每一步都能单独调，`pipeline()` 把它们串起来：

    chunking(folderpaths)  ->  list[ChunkDraft]
    merge(drafts)          ->  list[ChunkDraft]
    embedding(drafts)      ->  list[Chunk]
    store(chunks)          ->  None         <- 收口点：三张表一个事务

**只有 `store()` 同时看到三张表。** 它认得的三个存储各自只认识自己那张表；
扩展列的值由调用方通过 `fields_of` 给，本层不认识这些列的含义。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Literal

from simple_rag.chunking.structured_file.md_chunk_by_title import (
    chunk_by_title,
    merge_small_chunks,
)
from simple_rag.chunking.structured_file.types import (
    Chunk,
    ChunkDraft,
    render_text,
)
from simple_rag.db import Database
from simple_rag.embedding_api import TextPart
from simple_rag.repository.bm25 import BM25Index, create_bm25
from simple_rag.repository.chunk_store import (
    TABLE_NAME as DATA_TABLE,
    ChunkStore,
    Schema as ChunkSchema,
)
from simple_rag.repository.vec import (
    TABLE_NAME as VEC_TABLE,
    Schema as VecSchema,
    VecDB,
    vector_to_sqlite_vector,
)
from simple_rag.tokenization import Tokenizer

# 单次索引任务里同时在飞的 embedding 请求数。
#
# 实测（2026-10-02）：豆包按账户限流（429 `AccountRateLimitExceeded`），阈值在
# 大请求 170 条上下，**且不返回 `retry-after`**。并发 4 的多次探测没触发过，
# 8 一撞就整批失败 —— 所以压到 4。真撞上了由客户端的退避重试兜着
# （`DoubaoEmbeddingVision` 的 `max_retries`）。
_EMBED_CONCURRENCY = 4

# 扩展列的取值回调：一个 chunk 进，一份 `{列名: 值}` 出。
# 两张表各按自己的 Schema 去里面取自己声明过的列。
FieldsOf = Callable[[Chunk], Mapping[str, str | None]]


@dataclass
class RagIndexStrategy:
    """一次索引任务的通道配置：决定建哪些索引、各自怎么建。

    两个通道至少要开一个；开了通道就必须给全它需要的配置，否则对象构造不出来
    （校验见 `__post_init__`）。

    Attributes:
        use_vec: 是否启用向量索引通道。
        use_bm25: 是否启用 BM25 索引通道。
        chunk_schema: 数据表的形状，声明调用方要加的扩展列。
        min_tokens: 合并小块的下限，`merge()` 用。
        max_token: 单个 chunk 的 token 上限（软约束），`chunking()` 用。
        vec_schema: 向量表的形状（`dim` / `metric` / `fields` / `filterable`）。
            不启用向量通道时给 `None`。
        bm25_policy: BM25 用哪个实现，`"fts5"` 或 `"own_bm25"`；空串表示不选实现。
            两者都用注入的连接落盘，没有需要调用方另外给的持久化参数。
    """

    use_vec: bool
    use_bm25: bool
    chunk_schema: ChunkSchema
    min_tokens: int
    max_token: int
    vec_schema: VecSchema | None = None
    bm25_policy: Literal["fts5", "own_bm25", ""] = ""

    def __post_init__(self) -> None:
        """校验各字段的组合是否自洽，非法组合根本构造不出来。

        Raises:
            ValueError: 两个通道都关闭；开启向量通道但没给 `vec_schema`；
                开启 BM25 但没选策略。
        """
        if self.use_bm25 is False and self.use_vec is False:
            raise ValueError("不能设置全部索引通道都为False")
        if self.use_vec is True and self.vec_schema is None:
            raise ValueError("use_vec is True 但是没有设置 vec_schema")
        if self.use_bm25 is True and self.bm25_policy == "":
            raise ValueError("use_bm25 is True 但是没有选择策略")


def merge_chunks(min_tokens: int) -> Callable:
    """给切分函数套一层「切完顺手合并小块」。"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> list[ChunkDraft]:
            return merge_small_chunks(func(*args, **kwargs), min_tokens)

        return wrapper

    return decorator


class RagIndexingPipeline:
    """把「建立索引」这条链路内部的各环节整合到一起。

    配置由 `RagIndexStrategy` 给出，连接由 `Database` 提供并持有 —— 本类只借用
    连接，不负责它的生命周期。

    三张表（向量表 / 数据表 / BM25）在构造时一次性建好（``create_table`` /
    ``open`` 都幂等，重复构造同一个库不会报错），之后共用 ``db`` 的**同一个连接**
    —— 这正是一次入库能圈进一个事务的前提。
    """

    def __init__(
        self,
        index_strategy: RagIndexStrategy,
        db: Database,
        embedding_model,  # 类型待定
        tokenizer: Tokenizer,  # create_bm25 要它
    ) -> None:
        """初始化流水线。

        Args:
            index_strategy: 索引通道配置，决定建哪些索引。
            db: 连接的所有者。流水线从它取连接，关闭由创建方负责。
            embedding_model: 算向量用，需要 `await embed(parts, dimensions=...)`
                并返回带 `.dense` 的结果。
            tokenizer: 分词器，BM25 写入与查询共用。
        """
        self._strategy = index_strategy
        self._embedding_model = embedding_model
        self._conn = db.connection

        # 数据表是内容仓库，无论开哪个通道都要有
        self._chunk_store = ChunkStore(self._conn)
        self._chunk_store.create_table(index_strategy.chunk_schema)

        self._vec_db: VecDB | None = None
        if index_strategy.use_vec:
            self._vec_db = VecDB(self._conn)
            self._vec_db.create_table(index_strategy.vec_schema)

        self._bm25: BM25Index | None = None
        if index_strategy.use_bm25:
            # 两个实现都把语料落在注入的连接里，不需要别的持久化参数
            self._bm25 = create_bm25(
                index_strategy.bm25_policy,
                tokenizer=tokenizer,
                conn=self._conn,
            )
            self._bm25.open()

    def chunking(self, folderpaths: list[Path]) -> list[ChunkDraft]:
        """把若干目录下的文档切成块（只切，不合并 —— 合并交给 `merge()`）。

        递归找 `*.md`，按 **POSIX 形式的路径串**排序 —— 不用 `Path` 默认比较，
        它在 Windows 上忽略大小写、在 POSIX 上不忽略，同一批语料会切出不同的
        顺序。文件顺序进了 `chunk_id` 的路径集合，跨平台不一致就是主键漂移。

        `source_name` 取 `path.as_posix()`，所以**目录要传相对路径**：进
        `chunk_id` 的就是相对路径，主键才能跨机器可移植。传绝对路径不会报错，
        只是把 ID 绑死在这台机器上。
        """
        drafts: list[ChunkDraft] = []
        for folder in folderpaths:
            for path in sorted(
                Path(folder).rglob("*.md"), key=lambda p: p.as_posix()
            ):
                drafts.extend(
                    chunk_by_title(
                        path,
                        self._strategy.max_token,
                        0,
                        source_name=path.as_posix(),
                    )
                )
        return drafts

    def merge(self, chunk_drafts: list[ChunkDraft]) -> list[ChunkDraft]:
        """合并过小的块（阈值 `RagIndexStrategy.min_tokens`）。"""
        return merge_small_chunks(chunk_drafts, self._strategy.min_tokens)

    async def embedding(self, chunk_drafts: list[ChunkDraft]) -> list[Chunk]:
        """批量算向量，把 `ChunkDraft` 升成 `Chunk`。

        喂给模型的是 `render_text(segments, with_summary=True)` —— **带面包屑、
        也带摘要**。和入库的正文刻意不同源：摘要是模型生成的辅助信息，让向量
        更贴主题，但不该混进「这个 chunk 是什么」的正文里。

        并发度卡在 `_EMBED_CONCURRENCY`。返回顺序与入参一致（`gather` 保序）。
        """
        sem = asyncio.Semaphore(_EMBED_CONCURRENCY)

        async def one(text: str) -> list[float]:
            async with sem:
                result = await self._embedding_model.embed([TextPart(text)])
            return result.dense

        texts = [render_text(d.segments, with_summary=True) for d in chunk_drafts]
        vectors = list(await asyncio.gather(*(one(text) for text in texts)))
        return [
            Chunk(segments=draft.segments, tokens=draft.tokens, embedding_vec=vector)
            for draft, vector in zip(chunk_drafts, vectors)
        ]

    def store(
        self,
        chunks: list[Chunk],
        fields_of: FieldsOf | None = None,
    ) -> None:
        """收口点：三张表一个事务，全部成功或全部失败。

        每张表只拿自己 `Schema` 声明过的扩展列。`fields_of` 返回的映射里缺了
        某个声明过的列，会抛 `KeyError` —— 与各存储的 `insert` 一样严格，
        不静默写 `NULL`。`fields_of=None` 表示**所有**扩展列都写 `NULL`。

        **按 `chunk_id` 去重**：`chunk_id` 是内容寻址的，同一批里出现两条同 id
        只留第一条。这一步不能省 —— vec0 撞主键抛的是 `OperationalError`、
        `sqlite_errorcode` 还是通用的 `1`，只能靠字符串匹配判别。

        正文用 `render_text(segments)`（**不带摘要**），BM25 的 `text` 与它同源。
        算 embedding 用的是带摘要的文本，两者刻意不同：摘要是模型生成的辅助
        信息，不该混进「这个 chunk 是什么」的正文里。
        """
        strategy = self._strategy
        vec_fields = strategy.vec_schema.fields if strategy.vec_schema else ()

        vec_records: list[dict] = []
        data_records: list[dict] = []
        bm25_records: list[dict] = []
        seen: set[str] = set()

        for chunk in chunks:
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)

            # None 表示「没有扩展列」；给了回调却缺键是错误，交给 fields[...] 抛
            fields = dict(fields_of(chunk)) if fields_of is not None else None
            text = render_text(chunk.segments)

            if strategy.use_vec:
                vec_records.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "vector": chunk.embedding_vec,
                        **{
                            name: _pick(fields, name)
                            for name in vec_fields
                        },
                    }
                )
            data_records.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "content": text,
                    "sources": _sources_of(chunk),
                    **{
                        name: _pick(fields, name)
                        for name in strategy.chunk_schema.fields
                    },
                }
            )
            if strategy.use_bm25:
                bm25_records.append({"chunk_id": chunk.chunk_id, "text": text})

        def work() -> None:
            if vec_records:
                self._vec_db.insert(vec_records)
            if data_records:
                self._chunk_store.insert(data_records)
            if bm25_records:
                self._bm25.insert(bm25_records)

        self._in_transaction(work)

    async def pipeline(
        self,
        folderpaths: list[Path],
        fields_of: FieldsOf | None = None,
    ) -> None:
        """chunking → merge → embedding → store 串一遍。

        **不做「先删后写」** —— 重切一个目录时会不会留下旧 chunk，是调用方的
        判断：先 `delete_by_path` 哪些路径，再喂哪些目录进来，都由它决定。
        """
        chunks = await self.embedding(self.merge(self.chunking(folderpaths)))
        self.store(chunks, fields_of)

    def search_in_path(
        self, vector, k: int, path: str
    ) -> list[tuple[str, float]]:
        """在**该文件贡献的 chunk** 里做向量检索，返回 `(chunk_id, distance)`，近的在前。

        用**子查询**形态，不是参数列表 —— 实测（探针 25 复核）两者在 vec0 里走
        不同的路：

        - `chunk_id in (select ...)`：先按 path 筛出候选，再在候选里做 KNN。
          候选就算排不进全局前 k 名也照样召回
        - `chunk_id in (?, ?)`：先取全局最近的 k 条，再筛掉不在候选里的。
          **会静默少返回**（k=5 时明明有命中也返回空）

        返回的分数是距离（越小越像），与 `VecDB.search` 同一口径。

        Note:
            只返回 id 与距离，**不返回正文与字段** —— 那些在数据表里，
            拿到 id 再回 `ChunkStore` 取，是检索层一贯的路子。
        """
        if self._vec_db is None:
            raise ValueError("没启用向量通道，不能做向量检索")
        blob = vector_to_sqlite_vector(vector, self._strategy.vec_schema.dim)
        # 子查询整句交给 vec0，它才会当成约束去预过滤；拆成参数列表就退化成后置筛
        sql = (
            f"select chunk_id, distance from {VEC_TABLE} "
            f"where embedding match ? "
            f"and chunk_id in (select chunk_id from {DATA_TABLE} where path = ?) "
            f"and k = ?"
        )
        rows = self._conn.execute(sql, (blob, path, k)).fetchall()
        return [(row[0], row[1]) for row in rows]

    def delete_by_path(self, path: str) -> None:
        """按文件作废：sources 沾到该文件的 chunk 全部消失（三张表一起）。

        **不是「只删这个文件的行」** —— 跨文件合并出来的 chunk，正文是多个
        文件拼的；抽掉一个文件后，剩下那部分就不再是它声明的那个东西。
        所以它整个作废，连同它在数据表里其余来源的行。
        """
        chunk_ids = self._chunk_store.ids_by_path(path)
        if not chunk_ids:
            return

        def work() -> None:
            for chunk_id in chunk_ids:
                if self._vec_db is not None:
                    self._vec_db.delete(chunk_id)
                self._chunk_store.delete(chunk_id)
            if self._bm25 is not None:
                self._bm25.delete(chunk_ids)

        self._in_transaction(work)

    def _in_transaction(self, work: Callable[[], object]) -> None:
        """一个 `SAVEPOINT` 包住 `work`，失败回滚并原样抛出。

        用 `SAVEPOINT` 而不是 `BEGIN`：调用方可能还开着外层事务（比如要把
        别的表一起圈进来），那时 `BEGIN` 会报
        `cannot start a transaction within a transaction`。失败时除了
        `rollback to` 还必须 `release` —— 只回滚不释放会让事务一直挂着不提交，
        **而且不报错**。
        """
        conn = self._conn
        conn.execute("savepoint rag_index")
        try:
            work()
        except BaseException:
            conn.execute("rollback to rag_index")
            conn.execute("release rag_index")
            raise
        conn.execute("release rag_index")


def _pick(fields: Mapping[str, str | None] | None, name: str) -> str | None:
    """从回调给的映射里取一列。

    `fields` 为 `None`（没给回调）时全部写 `NULL`；给了回调就按列名取，
    取不到直接 `KeyError` —— 少给一列是调用方的疏漏，不该静默存成空。
    """
    if fields is None:
        return None
    return fields[name]


def _sources_of(chunk: Chunk) -> list[dict]:
    """把各段按来源文件合成行区间 —— 一个 `(chunk, path)` 一行。

    同一文件在 chunk 里的多个段取 min/max。段基本是连续切出来的，合成一段
    足够定位；真跨了行区间也只让范围略大，不会漏。
    """
    spans: dict[str, tuple[int, int]] = {}
    for seg in chunk.segments:
        lo, hi = spans.get(seg.file_path, (seg.start_line, seg.end_line))
        spans[seg.file_path] = (min(lo, seg.start_line), max(hi, seg.end_line))
    return [
        {"path": path, "start_line": lo, "end_line": hi}
        for path, (lo, hi) in spans.items()
    ]
