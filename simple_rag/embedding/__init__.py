"""`simple_rag.embedding` —— 索引构建的编排层。

对外两个名字：`RagIndexStrategy`（配置）与 `RagIndexingPipeline`（执行）。

本包是**唯一同时看到三张表**（向量表 / 数据表 / BM25）的地方 —— 收口点就是
`RagIndexingPipeline.store()`，一次入库圈在一个事务里。
"""

from .embedding import RagIndexingPipeline, RagIndexStrategy

__all__ = ["RagIndexingPipeline", "RagIndexStrategy"]
