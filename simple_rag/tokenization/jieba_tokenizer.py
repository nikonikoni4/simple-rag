"""jieba 分词器 —— 内置的默认实现。

jieba 0.42.1 实测。**版本参与 token 的生成** —— 升级 jieba 也应当视作换分词器
（词典变了，切分结果可能变），已索引的数据要重建。
"""

from __future__ import annotations

import jieba

from .base import drop_blank, register_tokenizer


@register_tokenizer("jieba")
class JiebaTokenizer:
    """结巴分词。

    构造时调一次 `jieba.initialize()` 预热词典（约 0.4 s，实测）。jieba 内部有
    `initialized` 短路，重复构造不会重复加载。
    """

    name = "jieba"

    def __init__(self) -> None:
        jieba.initialize()

    def tokenize(self, text: str) -> list[str]:
        """`drop_blank(jieba.lcut(text))`。

        **过滤放在这里，不放在调用方** —— 写入侧和查询侧共用这一个入口，
        调用方不可能忘。
        """
        return drop_blank(jieba.lcut(text))
