"""`simple_rag.rerank_api.base` —— 实现无关的公共层。

只放两样东西：统一接口（`Reranker`）与所有实现共用的输入输出形状。
**不放任何厂商逻辑** —— 每个实现各自内聚在自己的模块里。
"""

from .reranker import Reranker
from .types import RerankDocument, RerankHit

__all__ = ["Reranker", "RerankDocument", "RerankHit"]
