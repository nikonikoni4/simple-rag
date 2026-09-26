"""`simple_rag.embedding_api` —— 向量化服务商 API。

按服务商分文件，一个服务商一个模块。当前只有：

- `doubao` —— 豆包 `doubao-embedding-vision`（多模态，稠密 + 稀疏）
"""

from .doubao import (
    DoubaoEmbeddingResult,
    DoubaoEmbeddingVision,
    ImagePart,
    SparseEntry,
    TextPart,
    VideoPart,
)

__all__ = [
    "DoubaoEmbeddingResult",
    "DoubaoEmbeddingVision",
    "ImagePart",
    "SparseEntry",
    "TextPart",
    "VideoPart",
]
