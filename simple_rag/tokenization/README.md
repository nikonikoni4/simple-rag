# simple_rag/tokenization —— 可插拔分词

**为什么单独一层**：分词器与索引绑定 —— 换了分词器，已落盘的 token 串全部失效、
必须全库重建。它既不属于 `bm25`，将来也可能有别的地方要用，所以单独成包。

## 用法

```python
from simple_rag.tokenization import TokenizerFactory, to_document

tok = TokenizerFactory.create("jieba")

# 写入侧
tokens = tok.tokenize(原文)
bm25.insert([{"chunk_id": cid, "text": 原文}])   # 或自己拼 token 串：to_document(tokens)

# 查询侧 —— 必须用同一个分词器
bm25.search(查询词, k=10)
```

## 契约

`Tokenizer` 协议要求实现保证两件事，否则 token 串还原会错位：

| # | 保证 | 为什么 |
|---|---|---|
| 1 | 返回的 token **不含纯空白** | jieba 把空白本身切成独立 token（`'a b'` → `['a',' ','b']`，制表符 / 换行 / 全角空格同理，**实测**） |
| 2 | 返回的 token **不含内部空格** | token 串用空格拼接落盘，含空格会破坏还原 |

内置的 `JiebaTokenizer` 用 `drop_blank` 保证第 1 条。**过滤放在分词器里、不放在
调用方** —— 写入侧和查询侧共用这一个入口，调用方不可能忘。

实测：jieba **不产出「内部含空格」的 token**（含空白的永远是纯空白 token 本身），
所以空格可以安全地当拼接分隔符。

## ⚠️ 换分词器必须全库重建

写入侧和查询侧用了不同的分词器（**含同一分词器不同版本**）时，同一个词在两侧
不是同一个 token —— **结果是静默零召回，不报错**。

`jieba` 的版本也参与 token 生成（词典变了，切分结果可能变），
**升级 jieba 同样要视作换分词器**。

## 加一个新分词器

写一个类 + 一行注册：

```python
from simple_rag.tokenization import drop_blank, register_tokenizer


@register_tokenizer("pkuseg")
class PkusegTokenizer:
    name = "pkuseg"

    def tokenize(self, text: str) -> list[str]:
        return drop_blank(pkuseg.cut(text))
```

再让模块被 import 一次触发注册（在 `__init__.py` 里加一行 import），
就能 `TokenizerFactory.create("pkuseg")`。

**重复注册同一个名字报 `ValueError`** —— 静默覆盖会让「配置到底选了哪个」
变得不可知。

## 文件

| 文件 | 职责 |
|---|---|
| `__init__.py` | 对外出口：协议 + 工厂 + 2 个纯函数 + `JiebaTokenizer` |
| `base.py` | 注册表、`Tokenizer` 协议、`drop_blank` / `to_document` |
| `jieba_tokenizer.py` | jieba 实现 |
| `tests/` | 25 个用例。`python -m pytest simple_rag/tokenization` |

**注册表放在 `base.py` 而不是 `__init__.py`** —— 否则形成
`__init__ → jieba_tokenizer → __init__` 的循环 import。

## 依赖

`jieba` 是**硬依赖**（不是 optional extra）—— `__init__.py` 在 import 时加载
`jieba_tokenizer`，做成 extra 会让默认安装下 `import simple_rag.tokenization`
直接 `ModuleNotFoundError`。
