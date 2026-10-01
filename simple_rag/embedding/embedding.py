
"""索引构建的编排层：把「建索引」这条链路的内部环节整合到一起。

对外两个名字：`RagIndexStrategy` 是配置，`RagIndexingPipeline` 是执行。
"""

from functools import wraps
from pathlib import Path
from typing import Literal

from simple_rag.chunking.structured_file.types import ChunkDraft
from simple_rag.db import Database
from simple_rag.repository.chunk_store import Schema as ChunkSchema
from simple_rag.repository.vec import Schema as VecSchema
from simple_rag.repository.chunk_store import ChunkStore,ChunkRow
from simple_rag.repository.vec import VecDB
from simple_rag.config import RagIndexStrategy
from simple_rag.chunking.structured_file.md_chunk_by_title import merge_small_chunks
# 1. 创建数据库对象
# 2. 

from dataclasses import dataclass

@dataclass
class RagIndexStrategy:
    """一次索引任务的通道配置：决定建哪些索引、各自怎么建。

    两个通道至少要开一个；开了通道就必须给全它需要的配置，否则对象构造不出来
    （校验见 `__post_init__`）。

    Attributes:
        use_vec: 是否启用向量索引通道。
        use_bm25: 是否启用 BM25 索引通道。
        bm25_policy: BM25 用哪个实现，`"fts5"` 或 `"rank_bm25"`；空串表示不选实现。
        chunk_schema: 数据表的形状，声明调用方要加的扩展列。
        vec_schema: 向量表的形状（`dim` / `metric` / `fields` / `filterable`）。
            不启用向量通道时给 `None`。
        rank_bm25_store_path: `rank_bm25` 的持久化文件路径。这个实现没有别的
            落地方式，选它就必须给。
    """

    use_vec : bool
    use_bm25 : bool
    bm25_policy : Literal["fts5","rank_bm25",""] = "rank_bm25"
    chunk_schema : ChunkSchema 
    min_tokens :int 
    vec_schema : VecSchema | None 
    rank_bm25_store_path : Path | None = None
    
    def __post_init__(self):
        """校验各字段的组合是否自洽，非法组合根本构造不出来。

        Raises:
            ValueError: 两个通道都关闭；开启向量通道但没给 `vec_schema`；
                开启 BM25 但没选策略；选了 `rank_bm25` 但没给持久化路径。
        """
        if self.use_bm25 is False and self.use_vec is False:
            raise ValueError("不能设置全部索引通道都为False")
        elif self.use_vec is True and self.vec_schema is None:
            raise ValueError("use_vec is True 但是没有设置 vec_schema")
        elif self.use_bm25 is True and self.bm25_policy == "":
            raise ValueError("use_bm25 is True 但是没有选择策略")
        elif self.use_bm25 is True and self.bm25_policy == "rank_bm25" and self.rank_bm25_store_path is None :
            raise ValueError("已选择bm25索引方式未rank_bm25,但是没有设置数据存储地址")
def merag_chunks(min_tokens):
    def decorator(func):
        @wraps(func)
        def wrapper(*args,**kwargs):
            chunk_drafts = func(*args,**kwargs)
            return merge_small_chunks(chunk_drafts,min_tokens)
        return wrapper
    return decorator

class RagIndexingPipeline:
    """把「建立索引」这条链路内部的各环节整合到一起。

    配置由 `RagIndexStrategy` 给出，连接由 `Database` 提供并持有 —— 本类只借用
    连接，不负责它的生命周期。
    """

    def __init__(self,index_strategy:RagIndexStrategy,db:Database,embedding_model):
        """初始化流水线。

        Args:
            index_strategy: 索引通道配置，决定建哪些索引。
            db: 连接的所有者。流水线从它取连接，关闭由创建方负责。
        """
        

        pass
    def chunking(self,folderpaths:list[Path]):
        pass

    def merge(self,chunk_drafts:list[ChunkDraft]):
        """
        融合chunks
        """
        pass

    def embedding(self,):
        pass
    def store(self,):
        """
        
        """
        pass 

    def pipeline(self,):
        pass 