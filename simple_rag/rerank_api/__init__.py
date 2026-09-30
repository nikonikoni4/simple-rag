"""`simple_rag.rerank_api` —— 重排序服务商 API。

按服务商分文件，一个服务商一个模块。当前只有：

- `aliyun` —— 阿里云百炼 `qwen3.7-text-rerank`（异步，输出带 `chunk_id` 的排序，不含分数）

对外出口：

- `Reranker` —— 统一接口（`base/`），调用方只依赖它
- `AliyunReranker` —— 阿里云实现
- `RerankDocument` / `RerankHit` —— 输入输出形状（所有实现共用）
"""

from .aliyun import AliyunReranker
from .base import RerankDocument, RerankHit, Reranker

__all__ = [
    "AliyunReranker",
    "RerankDocument",
    "RerankHit",
    "Reranker",
]
