"""simple_rag.retrieval —— 检索层。

**不 import 子模块** —— `retrieval.py` 还是半成品，导入它会把 `embedding_api`
和网络客户端一起拉起来。

> 本文件的存在还让 `simple_rag.retrieval` 能被 `setuptools.find_packages` 找到。
> `dense/` 与 `sparse/` 是空目录，不补。
"""

__all__: list[str] = []
