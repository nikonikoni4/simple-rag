"""`simple_rag.tokenization` —— 可插拔的分词。

**为什么单独一层**：分词器与索引绑定 —— 换了分词器，已落盘的 token 串全部失效、
必须全库重建。它既不属于 `bm25`，将来也可能有别的地方要用，所以单独成包。

内置 `jieba`。加新分词器 = 写一个类 + 一行注册::

    @register_tokenizer("pkuseg")
    class PkusegTokenizer:
        name = "pkuseg"
        def tokenize(self, text: str) -> list[str]: ...

    TokenizerFactory.create("pkuseg")

⚠️ **换分词器必须全库重建。** 两侧不一致时同一个词不是同一个 token，
**结果是静默零召回、不报错**。
"""

from .base import (
    Tokenizer,
    TokenizerFactory,
    drop_blank,
    register_tokenizer,
    to_document,
)
from .jieba_tokenizer import JiebaTokenizer  # 触发注册（模块级副作用）

__all__ = [
    "JiebaTokenizer",
    "Tokenizer",
    "TokenizerFactory",
    "drop_blank",
    "register_tokenizer",
    "to_document",
]
