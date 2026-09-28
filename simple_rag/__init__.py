"""simple_rag —— RAG 模块。

各子包（`db` / `tokenization` / `repository` / `retrieval` / `embedding*` / `config`）
**互相独立**，本模块**不 import 任何子包** —— `import simple_rag` 只执行这一行，
不拖起 sqlite-vec、jieba 或网络客户端。

> 本文件的存在还有一个实际作用：没有它时 `simple_rag` 是 PEP 420 隐式命名空间包，
> `setuptools.find_packages` 会返回空列表，`pip install .` 装出来是个空包（实测）。
"""

__all__: list[str] = []
