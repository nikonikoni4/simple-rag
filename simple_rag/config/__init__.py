"""`simple_rag.config` —— 配置模块。

导入本包即触发 `.env` 加载（存在才加载），见 [config.py](config.py)。

对外出口：

- `DoubaoAPIConfig` —— 豆包 API 的 `api_base` / `api_key`
- `AliyunRerankAPIConfig` —— 阿里云百炼 rerank API 的 `api_base` / `api_key` / `model`
- `VecDBConfig` —— 向量库配置（占位）
"""

from .config import AliyunRerankAPIConfig, DoubaoAPIConfig, VecDBConfig

__all__ = ["AliyunRerankAPIConfig", "DoubaoAPIConfig", "VecDBConfig"]
