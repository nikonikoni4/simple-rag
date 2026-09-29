"""`simple_rag.repository` —— 存储层。两个**互不依赖**的子包：

- `vec`  —— sqlite-vec 的 `vec0` 表，向量相似度检索
- `bm25` —— 关键词检索，统一接口 + 两个实现（`fts5_bm25` / `okapi_bm25`），
  经 `create_bm25` 按实现名构造

**本包不导出任何名字。** 请从子包导入::

    from simple_rag.repository.vec import Schema, VecDB
    from simple_rag.repository.bm25 import create_bm25

不放转发 —— 同一个类有两条公开路径时，「该用哪个」和「改哪个」都会变得含糊。
"""

__all__: list[str] = []
