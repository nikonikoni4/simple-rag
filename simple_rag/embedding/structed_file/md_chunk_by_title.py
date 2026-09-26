# 按照标题进行切分md文档

"""
1. 加载file
1. 数据清洗
2. 递归构建md结构树(结构+每个节点的预估token数，需要一个计算token的工具函数)
3. 判断标题是否有错误(跳级，从# -> ### 等)->警告但继续按照策略进行
4. 切分chuak：(策略可选择，为了用于对比)
    1)表格代码等内容块查询,作为一个单独的子块
    2)文档超出退回
    
5. chunk并行Embedding 需要拼接前缀、特殊块的summary
6. 存入向量数据库(依赖注入)
"""


import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .types import ChunkDraft, MDFileHead, SpecialContent

# markdown 标题行：1~6 个 # 加空白再加标题文本
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

def _clean_file_content(content:str)->str:
    """清洗原文：去掉每行行尾空格，并把连续空行收敛成一行。

    Args:
        content: 待清洗的文本。

    Returns:
        清洗后的文本。**行数可能变少**（连续空行被吃掉），所以下游算出的行号
        是相对清洗后文本的，不是原文件行号。
    """
    lines = content.splitlines()
    cleaned = []
    for line in lines:
        line = line.rstrip()
        if line == "" and cleaned and cleaned[-1] == "":
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _count_token(text: str) -> int:
    """粗略估算 token 量，只用于装箱判断，不追求与真实分词器一致。

    Args:
        text: 待估算的文本。

    Returns:
        估算的 token 数。CJK 字符按 1 token，其余字符按约 4 字符 1 token。
        因此日文假名、韩文谚文会落到「非 CJK」那一边（当前语料是中文，可忽略）。
    """
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    return cjk + (len(text) - cjk) // 4


def _build_file_tree(clean_content:str)->MDFileHead: 
    """用「单遍扫描 + 栈」把标题结构建成树。

    Args:
        clean_content: 已清洗的 Markdown 全文。

    Returns:
        虚拟根节点。它的 `pref` 为空、`content` 是第一个标题之前的正文；
        `child_head` 是各一级标题（可以并列多个）。真实标题节点的 `pref` 必非空，
        所以「`pref == ""`」可以当作虚拟根的判据（前提是文本已清洗）。

    Note:
        标题跳级（`#` 直接到 `###`）不报错，按「就近挂到栈顶父节点」处理；
        代码围栏内的 `# 注释` 会被误判为标题，当前未处理；Setext 式标题
        （下一行跟 `===`）也不识别。
    """
    root = MDFileHead(pref="", content="", content_token=0)
    # 栈内元素为 (标题层级, 节点)，虚拟根层级为 0
    stack: list[tuple[int, MDFileHead]] = [(0, root)]
    body: list[str] = []  # 当前节点正在收集的正文行

    def flush() -> None:
        """把已收集的正文行追加到栈顶节点"""
        if not body:
            return
        node = stack[-1][1]
        text = "\n".join(body)
        node.content = f"{node.content}\n{text}" if node.content else text
        body.clear()

    for line in clean_content.splitlines():
        matched = _HEADING_RE.match(line)
        if matched is None:
            body.append(line)
            continue
        flush()
        level = len(matched.group(1))
        title = matched.group(2).strip()
        while stack[-1][0] >= level:
            stack.pop()
        parent = stack[-1][1]
        pref = f"{parent.pref} > {title}" if parent.pref else title
        node = MDFileHead(pref=pref, content=line, content_token=0)
        parent.child_head.append(node)
        stack.append((level, node))
    flush()

    _fill_tokens(root)
    return root


def _fill_tokens(node: MDFileHead) -> None:
    """后序遍历，自底向上回填 token 统计。

    必须在整棵树建好之后调用：父节点的 `child_token` 依赖所有子节点的
    `total_tokens`，而 `__post_init__` 在子节点建出来之前算不出来。

    Args:
        node: 子树根。原地修改，不返回新对象。
    """
    node.content_token = _count_token(node.content)
    node.child_token = 0
    for child in node.child_head:
        _fill_tokens(child)
        node.child_token += child.total_tokens
    node.total_tokens = node.content_token + node.child_token

# ---------------------------------------------------------------- 第 4 步：切分
#
# 顺序：
#   1) 定位行号 —— 前序累计每个节点占多少行，**行号在这里才确定**，不进 dataclass
#   2) 识别特殊块 —— 代码块 / 表格挂到节点的 special_content（事实层，与策略无关）
#   3) 装箱 —— 整棵子树装得下就一个 chunk；装不下就下钻，逐个节点往里加
#   4) 正文超限 —— 对正文切开，但切点避开代码块和表格


_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


@dataclass
class _Block:
    """正文里的一个单元。`code` / `table` 是硬边界，`text` 是软边界。

    行偏移是**节点内**的（0-based）；换算成全局行号要加上节点的起始行。

    Attributes:
        kind: 块类型。`code` / `table` 不可从中间切开，`text` 可以。
        start: 起始行偏移（闭区间，含）。
        end: 结束行偏移（闭区间，含）。
        tokens: 块的 token 量预估。
    """

    kind : Literal["code", "table", "text"]
    start : int  # 在所属节点 content 内的行偏移（0-based，闭区间）
    end : int
    tokens : int


@dataclass
class _Seg:
    """渲染片段。一个 chunk 由若干连续片段拼成。

    与 `_Block` 的分工：`_Block` 是「节点内的行偏移」，`_Seg` 是「已换算成全局
    行号、可以直接渲染的文本」。

    Attributes:
        pref: 该片段自己的面包屑，渲染时逐片段注入。
        text: 片段正文（可能只是节点 `content` 的一部分）。
        start_line: 起始行（闭区间，全局）。
        end_line: 结束行（闭区间，全局）。
        tokens: 片段 token 量。
        special_content: 落在本片段内的代码块 / 表格。
    """

    pref : str
    text : str
    start_line : int
    end_line : int
    tokens : int
    special_content : list[SpecialContent]


def _locate_spans(root: MDFileHead) -> dict[int, tuple[int, int]]:
    """前序遍历，算出每个节点占用的行区间（闭区间）。

    只累加 `content` 的行数，**不做文本搜索** —— 正文里出现两张一样的表格时，
    搜索会定位到错的那一张，而按阅读顺序累加不会。content 里保留着换行符，
    行数直接数得出来，所以也不需要额外的字段。

    行号相对**清洗后**的文本（`_clean_file_content` 会吃掉连续空行），不是原文件行号。

    key 用 `id(node)`：MDFileHead 是可变的 dataclass，`__hash__` 被置空，不能直接当键。

    Args:
        root: 虚拟根节点。

    Returns:
        `{id(节点): (起始行, 结束行)}`，闭区间，按前序顺序连续铺满全文。
        节点自身的区间是连续的（含其中的空行）；但 `_split_content` 产出的 chunk
        会跳过空行，所以 chunk 之间可能出现行号跳跃。
    """
    spans: dict[int, tuple[int, int]] = {}
    cursor = 0

    def walk(node: MDFileHead) -> None:
        """前序遍历，按 `content` 的行数推进光标并记录区间。"""
        nonlocal cursor
        width = node.content.count("\n") + 1 if node.content else 0
        spans[id(node)] = (cursor, cursor + width - 1)
        cursor += width
        for child in node.child_head:
            walk(child)

    walk(root)
    return spans


def _is_table_start(lines: list[str], i: int) -> bool:
    """判断第 `i` 行是不是一个 Markdown 表格的开头。

    要求两行同时成立：本行是表格行，且**下一行是分隔行**（`|---|:--:|`）。
    只看一行 `| a | b |` 不足以确定是表格 —— 正文里的竖线也会长这样。

    Args:
        lines: 行列表。
        i: 待判断的行下标。

    Returns:
        `True` 表示从 `lines[i]` 起是一个表格。
    """
    if _TABLE_ROW_RE.match(lines[i]) is None:
        return False
    return i + 1 < len(lines) and _TABLE_SEP_RE.match(lines[i + 1]) is not None


def _parse_blocks(lines: list[str]) -> list[_Block]:
    """把行列表切成块，供切分时判断「哪里不能切」。

    Args:
        lines: 行列表（通常是某个节点 `content` 的 `splitlines()`）。

    Returns:
        按出现顺序排列的块列表。规则：

        - 代码块：从围栏行吃到闭合围栏；围栏要同种符号且长度不小于开围栏，
          找不到闭合就吃到末尾
        - 表格：从表头行吃到连续表格行结束
        - 文本：空行是边界
        - 空行本身不产出块，所以它不属于任何 chunk
    """
    blocks: list[_Block] = []
    total = len(lines)
    i = 0
    while i < total:
        fence = _FENCE_RE.match(lines[i])
        if fence is not None:
            marker = fence.group(1)
            j = i + 1
            while j < total:
                closing = _FENCE_RE.match(lines[j])
                if (
                    closing is not None
                    and closing.group(1)[0] == marker[0]
                    and len(closing.group(1)) >= len(marker)
                ):
                    break
                j += 1
            end = min(j, total - 1)
            blocks.append(_Block("code", i, end, _count_token("\n".join(lines[i : end + 1]))))
            i = end + 1
        elif _is_table_start(lines, i):
            j = i
            while j + 1 < total and _TABLE_ROW_RE.match(lines[j + 1]) is not None:
                j += 1
            blocks.append(_Block("table", i, j, _count_token("\n".join(lines[i : j + 1]))))
            i = j + 1
        elif lines[i].strip() == "":
            i += 1
        else:
            j = i
            while (
                j + 1 < total
                and lines[j + 1].strip() != ""
                and _FENCE_RE.match(lines[j + 1]) is None
                and not _is_table_start(lines, j + 1)
            ):
                j += 1
            blocks.append(_Block("text", i, j, _count_token("\n".join(lines[i : j + 1]))))
            i = j + 1
    return blocks


def _fill_special(node: MDFileHead, spans: dict[int, tuple[int, int]]) -> None:
    """识别节点正文里的代码块 / 表格，挂到 `special_content` 上。

    这是**事实层** —— 与切分策略无关，换策略不用重算，可以单独测。

    Args:
        node: 子树根。原地写 `special_content`，并递归到所有子节点。
        spans: `_locate_spans` 的结果，用来把节点内偏移换算成全局行号。
    """
    lines = node.content.splitlines()
    base = spans[id(node)][0]
    node.special_content = [
        SpecialContent(
            start_line=base + blk.start,
            end_line=base + blk.end,
            content="\n".join(lines[blk.start : blk.end + 1]),
            content_type=blk.kind,
        )
        for blk in _parse_blocks(lines)
        if blk.kind in ("code", "table")
    ]
    for child in node.child_head:
        _fill_special(child, spans)


def _seg_of_node(node: MDFileHead, spans: dict[int, tuple[int, int]]) -> _Seg:
    """把一个节点**自己**的正文（不含子节点）包成渲染片段。

    Args:
        node: 目标节点。
        spans: `_locate_spans` 的结果。

    Returns:
        覆盖该节点整段 `content` 的片段。`special_content` 做了浅拷贝，
        避免之后被重新识别改写。
    """
    start, end = spans[id(node)]
    return _Seg(node.pref, node.content, start, end, node.content_token, list(node.special_content))


def _subtree_segs(node: MDFileHead, spans: dict[int, tuple[int, int]]) -> list[_Seg]:
    """把整棵子树按阅读顺序展开成片段列表。

    Args:
        node: 子树根。
        spans: `_locate_spans` 的结果。

    Returns:
        前序展开的片段列表。`content` 为空的节点不产出片段。
    """
    segs = [_seg_of_node(node, spans)] if node.content else []
    for child in node.child_head:
        segs.extend(_subtree_segs(child, spans))
    return segs


def _render(segs: list[_Seg]) -> ChunkDraft:
    """把若干连续片段拼成一个 chunk。

    面包屑按**每个片段各自**注入，所以多个标题合进同一个 chunk 也不会串。

    Args:
        segs: 按阅读顺序排列的连续片段，不能为空。

    Returns:
        拼好的 `ChunkDraft`。行区间取首尾片段；`tokens` 为各片段之和；
        `special_content` 汇总各片段里的代码块 / 表格。
    """
    parts = [f"{seg.pref}\n{seg.text}" if seg.pref else seg.text for seg in segs]
    return ChunkDraft(
        original_content="\n\n".join(parts),
        start_line=segs[0].start_line,
        end_line=segs[-1].end_line,
        tokens=sum(seg.tokens for seg in segs),
        special_content=[s for seg in segs for s in seg.special_content],
    )


def _split_content(node: MDFileHead, max_token: int, spans: dict[int, tuple[int, int]]) -> list[ChunkDraft]:
    """节点正文自己就超限时，按块装箱切成多段。

    切点只落在软边界（文本块之间）。代码块 / 表格**不从中间切开** —— 它自己
    就超限时，允许这一段超限，整体保留。

    Args:
        node: 正文超限的节点。
        max_token: 单段的 token 上限（软约束，会被不可切的块突破）。
        spans: `_locate_spans` 的结果。

    Returns:
        切好的 `ChunkDraft` 列表，按阅读顺序排列。每段都带上该节点的面包屑。
    """
    lines = node.content.splitlines()
    base = spans[id(node)][0]
    drafts: list[ChunkDraft] = []
    cur: list[_Block] = []
    cur_tokens = 0

    def flush() -> None:
        """把已攒的块结算成一段 chunk，并清空缓冲。"""
        nonlocal cur, cur_tokens
        if not cur:
            return
        start, end = cur[0].start, cur[-1].end
        line_start, line_end = base + start, base + end
        drafts.append(
            ChunkDraft(
                original_content=(f"{node.pref}\n" if node.pref else "")
                + "\n".join(lines[start : end + 1]),
                start_line=line_start,
                end_line=line_end,
                tokens=cur_tokens,
                special_content=[
                    s
                    for s in node.special_content
                    if s.start_line >= line_start and s.end_line <= line_end
                ],
            )
        )
        cur, cur_tokens = [], 0

    for blk in _parse_blocks(lines):
        if cur and cur_tokens + blk.tokens > max_token:
            flush()
        cur.append(blk)
        cur_tokens += blk.tokens
    flush()
    return drafts


def _pack(
    node: MDFileHead,
    max_token: int,
    spans: dict[int, tuple[int, int]],
    out: list[ChunkDraft],
) -> None:
    """从 `node` 往下装箱，把子树填成若干 chunk。

    规则：

    1. 整棵子树装得下 -> 直接一个 chunk
    2. 装不下 -> 先放本节点正文，再按阅读顺序逐个往里加子节点
    3. 加不下就结算当前 chunk，从下一个子节点重新开一个
    4. 子节点自己就超限 -> 下钻递归；到最下层正文还超限则交给 `_split_content`

    Args:
        node: 当前子树根。
        max_token: 单个 chunk 的 token 上限（软约束）。
        spans: `_locate_spans` 的结果。
        out: 结果累加到这里，按阅读顺序追加。原地修改，不返回。
    """
    if 0 < node.total_tokens <= max_token:
        out.append(_render(_subtree_segs(node, spans)))
        return

    buf: list[_Seg] = []
    buf_tokens = 0

    def flush() -> None:
        """把已攒的片段结算成一个 chunk，并清空缓冲。"""
        nonlocal buf, buf_tokens
        if buf:
            out.append(_render(buf))
            buf, buf_tokens = [], 0

    # 本节点正文在子节点之前
    if node.content:
        if node.content_token <= max_token:
            buf.append(_seg_of_node(node, spans))
            buf_tokens += node.content_token
        else:
            flush()
            out.extend(_split_content(node, max_token, spans))

    for child in node.child_head:
        if child.total_tokens > max_token:
            flush()
            _pack(child, max_token, spans, out)
            continue
        if buf and buf_tokens + child.total_tokens > max_token:
            flush()
        buf.extend(_subtree_segs(child, spans))
        buf_tokens += child.total_tokens
    flush()


def _cut_chunks(tree: MDFileHead, max_token: int) -> list[ChunkDraft]:
    """按标题切分整棵树。

    Args:
        tree: `_build_file_tree` 产出的树（虚拟根）。
        max_token: 单个 chunk 的 token 上限（软约束）。

    Returns:
        按阅读顺序排列的 `ChunkDraft` 列表。整篇装得下时只返回一个。

    Note:
        可能产出 `tokens == 0` 的碎片（例如单独一行的零宽字符），当前**未过滤** ——
        需要的话在这里加一道筛选。
    """
    spans = _locate_spans(tree)
    _fill_special(tree, spans)
    out: list[ChunkDraft] = []
    _pack(tree, max_token, spans, out)
    return out




def chunk_by_title(file_path:Path,max_token,start_line,end_line=None)->list[ChunkDraft]:
    """按标题把一个 Markdown 文件切成 chunk 草稿。

    流程：读取文件 -> 按行切片 -> 清洗 -> 建标题树 -> 切分。

    Args:
        file_path: Markdown 文件路径，按 UTF-8 读取。
        max_token: 单个 chunk 的 token 上限（软约束，不可切的块会突破）。
        start_line: 起始行（0-based，左闭）。
        end_line: 结束行（0-based，**右开**）；`None` 表示读到文件末尾。

    Returns:
        按阅读顺序排列的 `ChunkDraft` 列表。

    Note:
        第 4 步之后的行号是相对**清洗后**文本的；叠加这里的切片偏移后，它已经
        不等于原文件行号。要精确回溯原文需要额外维护映射 —— `_clean_file_content`
        会吃掉连续空行，所以映射不是简单加一个偏移量。
    """
    # 1. 读取文件
    content = file_path.read_text(encoding='utf-8')
    content = "\n".join(content.splitlines()[start_line:end_line])
    # 2. 清洗:当前仅仅清洗空格和多行(原样是当前所设计到的文档基本收都是AI或我写的,比较干净)
    # 后续若出现其他文档,再添加别的清洗策略再simple_rag\embedding\clean_pipeline
    clean_content = _clean_file_content(content)
    # 3. 构建文档树
    filetree = _build_file_tree(clean_content)
    # 4. 切分
    return _cut_chunks(filetree, max_token)

if __name__ =="__main__":
    file_path = Path(r"D:\desktop\软件开发\RAG\data\lifeprismData\diary\2025\01\2025-01-16.md")
    content = file_path.read_text(encoding='utf-8')
    print(content)