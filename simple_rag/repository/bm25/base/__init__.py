"""`simple_rag.repository.bm25.base` —— 实现无关的公共层。

只放两样东西：统一接口（`BM25Index`）与所有实现共用的输出形状 / 校验。
**不放任何存储逻辑** —— 每个实现各自内聚在自己的子包里。
"""

from .index import BM25Index
from .types import BM25SearchResult
from .validation import check_k, check_record, prepare_rows

__all__ = ["BM25Index", "BM25SearchResult", "check_k", "check_record", "prepare_rows"]
