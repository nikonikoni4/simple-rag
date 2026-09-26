"""跨步骤传递的类型定义。

切分（`md_chunk_by_title`）、embedding、入库之间流通的对象都定义在这里，
实现留在各自的模块：

- `MDFileHead` —— 标题树的节点，切分阶段的中间结构
- `SpecialContent` —— 需保护、不能从中间切开的块（代码 / 表格）
- `Segment` —— chunk 里的一段：纯正文 + 它自己的面包屑 + 行区间
- `ChunkDraft` —— 切分产物，还没有 `embedding_vec`（第 5 步填）
- `Chunk` —— 最终产物，带 `embedding_vec`，可直接交给 `VecStore`

**面包屑和摘要都不写进 `Segment.text`**，它们是独立字段：存的时候分开存，
要用的时候（算 embedding、给 agent 看）才由 `Segment.render` 拼起来。
这样调用方拿到的是「原文 + 可选的附加信息」，而不是一段被改写过、还原不回去的文本
—— 给不给 agent 摘要、给不给面包屑，是调用方的决定，不是切分这一步能替它做的。
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
        summary: 块摘要。第 4 步按需生成，`None` 表示没生成或生成失败。
    """

    start_line : int
    end_line : int
    content : str
    content_type : Literal["code","table"] # 如果后续识别更多的内容快在添加
    summary : str|None = None  # 摘要，第 4 步按需填


@dataclass
class Segment:
    """chunk 里的一段：某个节点**自己**的一段正文，外加它的定位信息。

    切分不会把段拼成一个大字符串，而是保留成列表，就是为了让下面这些信息都活着：
    面包屑分得清、原文还原得回、摘要能按行号插回原位。

    Attributes:
        pref: 面包屑，从根标题一路拼到当前标题（含当前标题），分隔符 `" > "`。
            该段不属于任何标题时为空串。
        text: **纯原文片段**，不含面包屑、不含摘要。
        start_line: 起始行（闭区间，全局，相对清洗后文本）。
        end_line: 结束行（闭区间，全局）。
        tokens: 该段的 token 量预估。
        special_content: 落在本段行区间内的代码块 / 表格。摘要挂在这些对象上。
    """

    pref : str
    text : str
    start_line : int
    end_line : int
    tokens : int
    special_content : list[SpecialContent] = field(default_factory=list)

    def render(self, with_summary: bool = False) -> str:
        """拼成可以直接用的文本：面包屑 + 正文（可选：把特殊块摘要插进正文）。

        摘要插在它那块**之前**、原块原样保留 —— 摘要充当语义小标题，让这块能被
        「问出来」，同时不丢原文的数字与细节。

        定位靠**行号**，不靠搜文本：正文里出现两张一样的表格时，搜索会定位到错的
        那一张，而按行号算偏移不会。

        Args:
            with_summary: 是否插入摘要。算 embedding 时用 `True`；只想看原文时
                保持 `False`。没有 `summary` 的块（没生成 / 生成失败）跳过不插。

        Returns:
            拼好的文本。`pref` 为空时不额外加空行。
        """
        body = self.text
        if with_summary:
            inserts: dict[int, list[str]] = {}
            for block in self.special_content:
                if block.summary is None:
                    continue
                offset = block.start_line - self.start_line
                if 0 <= offset < len(self.text.splitlines()):
                    inserts.setdefault(offset, []).append(block.summary)
            if inserts:
                lines: list[str] = []
                for offset, line in enumerate(self.text.splitlines()):
                    lines.extend(inserts.get(offset, ()))
                    lines.append(line)
                body = "\n".join(lines)
        return f"{self.pref}\n{body}" if self.pref else body


def render_text(segments: Sequence[Segment], *, with_summary: bool = False) -> str:
    """把若干段拼成可以直接用的文本。

    段之间用空行隔开，和原文档里段与段之间的样子一致。

    Args:
        segments: 按阅读顺序排列的片段。
        with_summary: 透传给 `Segment.render`。

    Returns:
        拼好的文本。`segments` 为空时返回空串。
    """
    return "\n\n".join(seg.render(with_summary) for seg in segments)


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
        segments: 按阅读顺序排列的片段。**面包屑在 `Segment.pref` 上，不在 text 里**，
            所以一个 chunk 里混着多个标题的段也不会串，而且原文可以原样拿回来。
        file_path: 来源文件路径，**原样保留调用方传进来的值**。它直接参与
            `chunk_id`，所以传相对路径（相对语料根）才能让 ID 跨机器可移植；
            传绝对路径会把 ID 绑死在这台机器上。
        tokens: 各段 token 之和。自身超限的代码块 / 表格会超过 `max_token`。
        start_line: 起始行（闭区间，全局），由 `segments` 首段推出。
        end_line: 结束行（闭区间，全局），由 `segments` 末段推出。
        special_content: 落在本 chunk 内的代码块 / 表格（带摘要），由各段汇总。
            只是方便取用，不额外存一份 —— 真正的归属在 `Segment.special_content` 上。
    """

    segments : list[Segment]
    file_path : str
    tokens : int
    start_line : int = None # post_init
    end_line : int = None # post_init
    special_content : list[SpecialContent] = field(default_factory=list) # post_init

    def __post_init__(self):
        """行区间与 `special_content` 都从 `segments` 推导，不接受外部传入。"""
        self.start_line = self.segments[0].start_line
        self.end_line = self.segments[-1].end_line
        self.special_content = [
            block for seg in self.segments for block in seg.special_content
        ]


@dataclass
class Chunk:
    """最终产物：一段可入库的 chunk。可直接交给 `VecStore.insert`。

    Attributes:
        segments: 同 `ChunkDraft.segments`。
        file_path: 来源文件路径。**直接参与 `chunk_id`** —— 传相对路径才能跨机器
            可移植，传绝对路径会把主键绑死在这台机器上。
        tokens: token 量预估。
        embedding_vec: 向量。构造时可传 `list[float]`（API 常见格式）、`tuple[float]`
            或 `ndarray`；`__post_init__` 统一转成 float32 的 ndarray。
        special_content: 落在本 chunk 内的代码块 / 表格。**摘要挂在这里**，不在
            `segments` 里 —— 所以入库之后 agent 仍然能单独拿到摘要。
        start_line: 起始行（闭区间），由 `segments` 首段推出。**只是参考信息，不再是
            唯一标识** —— 同一行被切成多段时，几段的行区间会完全相同。
        end_line: 结束行（闭区间），由 `segments` 末段推出。同上，仅作参考。
        chunk_id: 主键，**按内容寻址**：`H(file_path + content_hash)`。
            内容一样就是同一条记录（不必存两份），内容变了主键就变。
            路径参与主键，所以同内容不同文件仍是两条 —— 换来的好处是「按 `file_path`
            删」这种更新方式不会误删别的文件；若改成全局内容寻址，跨文件重复的内容
            只留一行、且只记一个路径，按文件增量删时会把它**静默删掉**。

            ⚠️ 因此**入库前必须先按 `chunk_id` 去重**（一批内和库里已有的都要）。
            同一批里出现两条同 id，第二条会撞 `UNIQUE` 约束；而 vec0 抛的是
            `OperationalError`、`sqlite_errorcode` 是通用的 `1`，**只能靠字符串匹配
            判别**，撞了很难查。
        content_hash: `render_text(segments)`（面包屑 + 正文，`with_summary=False`）的
            哈希。**和 `chunk_id` 的区别是它不含路径**，所以能回答「这段内容在库里
            出现过吗」（例如跨文件复用向量、省一次 API 调用）。

            两者都**不含摘要**：摘要是模型生成的、本身不确定，算进去会让每次重跑都
            全量重嵌（主键也跟着变，旧行全成孤儿）。代价是摘要变了两者都不变，
            调用方需要自己判断要不要重嵌。
        parent_id: 预留，将来做 parent 召回时使用。
    """

    segments : list[Segment]
    file_path : str
    tokens : int
    # 入参和持有都用这个联合类型：API 返回 list[float]，本地模型给 ndarray，
    # 两者都能直接交给 VecStore（它的入参就是 Sequence[float] | ndarray）。
    # __post_init__ 会把 list 转成 ndarray 省内存，所以构造完运行期一定是 ndarray
    embedding_vec: Sequence[float] | np.ndarray # post_init
    special_content : list[SpecialContent] = field(default_factory=list) # post_init
    start_line : int = None # post_init
    end_line : int = None # post_init
    chunk_id : str = None # post_init
    content_hash : str = None # post_init
    parent_id : str = None # 预留,若后面要做parent召回时使用

    def __post_init__(self):
        """推导行区间、`special_content`、`content_hash` 与 `chunk_id`，并统一向量 dtype。"""
        # list[float] -> ndarray：1536 维从 ~48KB 降到 ~6KB
        self.embedding_vec = np.asarray(self.embedding_vec, dtype=np.float32)
        self.start_line = self.segments[0].start_line
        self.end_line = self.segments[-1].end_line
        self.special_content = [
            block for seg in self.segments for block in seg.special_content
        ]
        # 去重指纹：喂给 embedding 的文本没变 -> hash 不变，用来决定要不要重算向量。
        # 摘要不算在内，理由见类 docstring
        self.content_hash = hashlib.blake2b(
            render_text(self.segments).encode("utf-8"), digest_size=16
        ).hexdigest()
        # 主键按**内容**寻址：文件路径 + 内容指纹 -> 同一内容 + 同一文件就是同一条记录。
        # 行号不参与 —— 位置会随文档增删整体平移，拿它当身份只会让「内容没变也换 ID」。
        # 内容改由调用方按 file_path 清掉旧行（见类 docstring）。
        # \x00 当分隔符：路径和十六进制指纹里都不可能含它，避免拼接歧义
        key = f"{self.file_path}\x00{self.content_hash}"
        self.chunk_id = hashlib.blake2b(key.encode("utf-8"), digest_size=16).hexdigest()
