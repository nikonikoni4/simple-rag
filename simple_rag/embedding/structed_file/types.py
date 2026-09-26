"""跨步骤传递的类型定义。

切分（`md_chunk_by_title`）、embedding、入库之间流通的对象都定义在这里，
实现留在各自的模块：

- `MDFileHead` —— 标题树的节点，切分阶段的中间结构
- `SpecialContent` —— 需保护、不能从中间切开的块（代码 / 表格）
- `ChunkDraft` —— 切分产物，还没有 `embedding_vec`（第 5 步填）
- `Chunk` —— 最终产物，带 `embedding_vec`，可直接交给 `VecStore`
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class SpecialContent:
    """节点正文里需要保护、不能从中间切开的块（代码 / 表格）。

    行号相对**清洗后**的文本，在切分时由前序累计得出。

    Attributes:
        start_line: 起始行（闭区间，含）。
        end_line: 结束行（闭区间，含）。
        content: 块的原始文本，不做任何改写。
        content_type: 块类型。目前只识别 `"code"` 与 `"table"`。
        summary: 块摘要。第 5 步按需生成，`None` 表示不生成。
    """

    start_line : int
    end_line : int
    content : str
    content_type : Literal["code","table"] # 如果后续识别更多的内容快在添加
    summary : str|None = None  # 摘要，第 5 步按需填


@dataclass
class MDFileHead:
    """Markdown 标题树的一个节点（`#` ~ `######`）。

    只承载结构与 token 统计，**不含行号** —— 行号在切分时才由 `_locate_spans` 算出。

    Attributes:
        pref: 面包屑，从根标题一路拼到当前标题（含当前标题），分隔符 `" > "`。
        content: 当前标题下的正文，首行是标题行本身，**不含子标题及其内容**。
        content_token: `content` 的 token 量预估（含标题行）。
        child_token: 所有子孙节点的 token 量之和。
        child_head: 子节点列表，按原文档出现顺序排列。
        total_tokens: `content_token + child_token`，由 `__post_init__` 算出。
        special_content: 从 `content` 识别出的代码块 / 表格，切分时由 `_fill_special` 填。
    """

    pref : str # 前缀标题（包含当前标题）
    content : str # 当前标题下的正文，不含子标题和其内容，不许是原封不动的原文，不能将pref，
    content_token : int # 正文的token量预估（包含标题）
    child_token : int =0 # 下面所有子标题的token量预估（包含标题）
    child_head : list["MDFileHead"] = field(default_factory=list) # 子节点
    total_tokens : int =0 #  正文 + child_token
    special_content : list[SpecialContent] = field(default_factory=list) # 代码、表格等需保护不受截断的块

    def __post_init__(self):
        """`total_tokens` 由 `content_token + child_token` 推导，不接受外部传入。"""
        self.total_tokens = self.content_token + self.child_token


@dataclass
class ChunkDraft:
    """一段切分结果。此时还没有 `embedding_vec`，由第 5 步并行 embedding 填。

    Attributes:
        original_content: 面包屑 + 正文，将来直接作为 `Chunk.original_content`。
            **不含摘要** —— 摘要单独存字段，只在算 embedding 时才拼接。
        start_line: 起始行（闭区间，相对清洗后文本）。
        end_line: 结束行（闭区间，相对清洗后文本）。
        tokens: 各片段 token 之和。自身超限的代码块 / 表格会超过 `max_token`。
        special_content: 落在这个 chunk 内的代码块 / 表格，供第 5 步拼摘要。
    """

    original_content : str  # 面包屑 + 正文，将来是 Chunk.original_content
    start_line : int
    end_line : int
    tokens : int
    special_content : list[SpecialContent] = field(default_factory=list)


@dataclass
class Chunk:
    """最终产物：一段可入库的 chunk。可直接交给 `VecStore.insert`。

    Attributes:
        start_line: 起始行（闭区间）。
        end_line: 结束行（闭区间）。
        file_name: 来源文件名。**直接参与 `chunk_id`** —— 若不含路径，不同目录的
            同名文件会算出同一个 ID。
        original_content: chunk 原文（面包屑 + 正文）。`content_hash` 与
            `embedding_vec` 都以它为基准，所以**不能塞摘要** —— 摘要单独存字段，
            只在算 embedding 时拼接。
        tokens: token 量预估。
        embedding_vec: 向量。构造时可传 `list[float]`（API 常见格式）、`tuple[float]`
            或 `ndarray`；`__post_init__` 统一转成 float32 的 ndarray。
        chunk_id: 主键，位置寻址（`file_name` + 行区间）的确定性哈希。
        content_hash: `original_content` 的哈希，用于判断内容有没有变（去重 / 增量）。
        parent_id: 预留，将来做 parent 召回时使用。
    """

    start_line : int
    end_line : int
    file_name : str 
    original_content : str 
    tokens : int 
    # 入参和持有都用这个联合类型：API 返回 list[float]，本地模型给 ndarray，
    # 两者都能直接交给 VecStore（它的入参就是 Sequence[float] | ndarray）。
    # __post_init__ 会把 list 转成 ndarray 省内存，所以构造完运行期一定是 ndarray
    embedding_vec: Sequence[float] | np.ndarray # post_init
    chunk_id : str = None # post_init
    content_hash : str = None # post_init
    parent_id : str =None # 预留,若后面要做parent召回时使用

    def __post_init__(self):
        """把向量统一成 float32 ndarray，并算出 `content_hash` 与 `chunk_id`。"""
        # list[float] -> ndarray：1536 维从 ~48KB 降到 ~6KB
        self.embedding_vec = np.asarray(self.embedding_vec, dtype=np.float32)
        # 去重指纹：内容没变 -> hash 不变，用来决定要不要重算 embedding
        self.content_hash = hashlib.blake2b(
            self.original_content.encode("utf-8"), digest_size=16
        ).hexdigest()
        # 主键走位置寻址：同一文件 + 同一行区间 -> 同一 ID。
        # 重跑幂等（长不出重复），内容改动也不会让旧 ID 变成孤儿。
        # \x00 当分隔符：文件名和行号里都不可能含它，避免拼接歧义
        key = f"{self.file_name}\x00{self.start_line}\x00{self.end_line}"
        self.chunk_id = hashlib.blake2b(key.encode("utf-8"), digest_size=16).hexdigest()
