"""配置加载。

导入本模块时，把项目根目录的 `.env` 读进环境变量；**文件不存在就跳过**，
此时只能靠构造配置类时显式传值。

配置取值的优先级：

1. 构造时传入的
2. 环境变量里的
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# 项目根目录的 .env（simple_rag/config/config.py -> 上两级）
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)


class DoubaoAPIConfig:
    """豆包（火山方舟）API 的接入配置。

    每个字段的优先级：构造时传入 > 环境变量。环境变量带服务商前缀，
    避免和系统里同名的通用变量（如 `API_KEY`）撞上。

    | 字段 | 环境变量 |
    |---|---|
    | `api_base` | `DOUBAO_API_BASE` |
    | `api_key` | `DOUBAO_API_KEY` |
    | `model` | `DOUBAO_EMBEDDING_MODEL_ID` |
    """

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_base = api_base or os.environ.get("DOUBAO_API_BASE")
        self.api_key = api_key or os.environ.get("DOUBAO_API_KEY")
        self.model = model or os.environ.get("DOUBAO_EMBEDDING_MODEL_ID")


class AliyunRerankAPIConfig:
    """阿里云百炼 rerank API 的接入配置。

    每个字段的优先级：构造时传入 > 环境变量。环境变量带服务商前缀，
    避免和系统里同名的通用变量（如 `API_KEY`）撞上。

    | 字段 | 环境变量 |
    |---|---|
    | `api_base` | `ALY_RERANK_BASE_URL` |
    | `api_key` | `ALY_API_KEY` |
    | `model` | `ALY_RERANK_API_MODEL` |

    注意 `api_base` 是**完整的原生 URL**（形如
    `https://<workspace-id>.cn-beijing.maas.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank`），
    直接 POST，不要再拼路径 —— 与豆包的「host + 拼路径」用法不同。
    """

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_base = api_base or os.environ.get("ALY_RERANK_BASE_URL")
        self.api_key = api_key or os.environ.get("ALY_API_KEY")
        self.model = model or os.environ.get("ALY_RERANK_API_MODEL")

