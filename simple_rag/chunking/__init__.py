"""simple_rag.chunking —— 语料切分与清洗。

**不 import 子包** —— `clean_pipeline` 与 `structured_file` 互相独立，用哪个导哪个。

> 本文件的存在还让 `simple_rag.chunking.*` 能被 `setuptools.find_packages` 找到 ——
> 没有它时整棵子树都打不进 wheel（`find_packages` 要求**每一层**都有 `__init__.py`）。
"""

__all__: list[str] = []
