"""重排序接口的输入输出形状。**所有实现共用，与厂商无关。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RerankDocument:
    """送排的一个候选：外部标识 + 正文。

    `chunk_id` 由调用方给定（通常来自检索层结果），实现必须**原样透传**到输出。
    """

    chunk_id: str
    text: str


@dataclass(frozen=True)
class RerankHit:
    """重排后的一条。`rerank` 返回列表的**顺序即相关度顺序（最相关在前）**。

    与检索层 `RetrievalHit` 的「score 越小越相关」不同，这里**没有分数** ——
    排序就是全部信息（要分数的厂商差异被接口挡在实现内部）。
    """

    chunk_id: str
    text: str
