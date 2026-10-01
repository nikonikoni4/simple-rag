# 按照标题进行切分md文档

"""
1. 加载file
1. 数据清洗
2. 递归构建md结构树(结构+每个节点的预估token数，需要一个计算token的工具函数)
3. 判断标题是否有错误(跳级，从# -> ### 等)->警告但继续按照策略进行
4. 对文档中的表格/代码块并发总结(总结函数由调用方注入，单个失败只 warning)
5. 切分策略
1）有标题的情况：
    “添加节点” ： 添加该节点的正文+所有子节点内容
    “添加正文”：只添加该节点的正文
    a. 从最顶层开始，判断相加是否超过max_token: 
        递归添加过程
        def 正文切割方法():... 
        chunk_list = []
        cur_token = 0
        def 切割(file_head:MDFileHead):
            nonlocal current_token
            if not MDFileHead:
                return 
            # 先添加正文
            cur_token += file_head.
        直到
"""


import logging
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .types import ChunkDraft, MDFileHead, Segment, SpecialContent

# markdown 标题行：1~6 个 # 加空白再加标题文本
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
# CJK 字符（`_count_token` 按它区分计费档位，单独抽出来给逐行累加复用）
_CJK_RE = re.compile(r"[一-鿿]")

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
    cjk = len(_CJK_RE.findall(text))
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

    Note:
        只统计 `content`，**不含特殊块摘要** —— 摘要是第 4 步才生成的独立字段。
        理由见 `chunk_by_title` 的 Note。
    """
    node.content_token = _count_token(node.content)
    node.child_token = 0
    for child in node.child_head:
        _fill_tokens(child)
        node.child_token += child.total_tokens
    node.total_tokens = node.content_token + node.child_token


# --------------------------------------------------- 第 4 步：识别特殊块 + 并发总结
#
# 顺序：
#   1) 定位行号 —— 前序累计每个节点占多少行，**行号在这里才确定**，不进 dataclass
#   2) 识别特殊块 —— 代码块 / 表格挂到节点的 special_content（事实层，与策略无关）
#   3) 并发总结 —— 把待总结的块交给调用方注入的 summary_func，单个失败只 warning


_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
# 句末标点：一行自己超预算时就近切在它们后面
_PUNCT_RE = re.compile(r"[。！？；!?;]")

# 并发总结的线程数。总结是网络 IO，所以用线程而不是进程。写死是因为
# 「缺了它模块照样能工作」—— 真需要调并发是调用方的事。
_SUMMARY_MAX_WORKERS = 4


@dataclass
class _Block:
    """正文里的一个单元。`code` / `table` 是硬边界，`text` 是软边界。

    行偏移是**节点内**的（0-based）；换算成全局行号要加上节点的起始行。

    Attributes:
        kind: 块类型。`code` / `table` 不可从中间切开，`text` 可以。
        start: 起始行偏移（闭区间，含）。
        end: 结束行偏移（闭区间，含）。
    """

    kind : Literal["code", "table", "text"]
    start : int  # 在所属节点 content 内的行偏移（0-based，闭区间）
    end : int


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
            blocks.append(_Block("code", i, end))
            i = end + 1
        elif _is_table_start(lines, i):
            j = i
            while j + 1 < total and _TABLE_ROW_RE.match(lines[j + 1]) is not None:
                j += 1
            blocks.append(_Block("table", i, j))
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
            blocks.append(_Block("text", i, j))
            i = j + 1
    return blocks


def _fill_special(node: MDFileHead, spans: dict[int, tuple[int, int]]) -> None:
    """识别节点正文里的代码块 / 表格，挂到 `special_content` 上。

    这是**事实层** —— 与切分策略无关，换策略不用重算，可以单独测。

    Args:
        node: 子树根。原地写 `special_content`，并递归到所有子节点。
        spans: `_locate_spans` 的结果，用来把节点内偏移换算成全局行号。

    Note:
        本函数是**整段重写** `node.special_content`，所以对同一棵树再调一次会把
        已经写好的 `summary` 全部抹掉。总结只在这里做一次，下游（切分）请直接复用
        结果，不要重新 fill。
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


def _collect_specials(root: MDFileHead) -> list[SpecialContent]:
    """前序遍历整棵树，收集**还没有摘要**的特殊块（表格 / 代码块）。

    这是「先找出需要总结的」那一步，在主线程里一次收完，不占线程池。

    Args:
        root: 虚拟根节点，`special_content` 需已由 `_fill_special` 填好。

    Returns:
        按阅读顺序排列的 `SpecialContent` 列表。返回的是**原对象**而非拷贝 ——
        并发总结时直接原地写回它们的 `summary`。
    """
    out: list[SpecialContent] = []

    def walk(node: MDFileHead) -> None:
        """前序收集本节点的待总结块，再递归子节点。"""
        out.extend(block for block in node.special_content if block.summary is None)
        for child in node.child_head:
            walk(child)

    walk(root)
    return out


def _summarize_specials(
    blocks: list[SpecialContent], summary_func: Callable[[str], str]
) -> None:
    """并发调用 `summary_func`，把结果写回各块的 `summary`。

    `with` 退出时会 `shutdown(wait=True)`，所以所有块跑完才返回 —— 也就是
    「并发执行，然后 wait」。写回的是各块自己的对象，**与完成顺序无关**，
    因此结果和串行执行完全一致。

    单个块失败只记 warning，不向外抛：一批里坏掉一个不该让整篇文档失败。
    该块的 `summary` 保持 `None`，其余块不受影响。

    Args:
        blocks: `_collect_specials` 的产物，原地写 `summary`。
        summary_func: 调用方注入的总结函数，入参是块的原文，返回摘要文本。
    """
    if not blocks:
        return
    workers = min(_SUMMARY_MAX_WORKERS, len(blocks))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(summary_func, block.content): block for block in blocks}
        for future in as_completed(futures):
            block = futures[future]
            try:
                block.summary = future.result()
            except Exception:
                # 只 warning 不抛：异常本身可预期（网络抖动 / 内容触发风控等），
                # 一个块坏掉不该让整篇文档总结失败。
                logging.warning(
                    "特殊块总结失败，已跳过（content_type=%s, 行 %d-%d）",
                    block.content_type,
                    block.start_line,
                    block.end_line,
                    exc_info=True,
                )


# 重叠区占 max_token 的比例。注释里说取 10%~20%,这里取中间值。
_OVERLOP_RATIO = 0.15


def _seg_of_node(
    node: MDFileHead, spans: dict[int, tuple[int, int]], file_path: str
) -> Segment:
    """把一个节点**自己**的正文（不含子节点）包成 `Segment`。

    `special_content` 用**节点上原来那批对象**（第 4 步填的），不重新造 ——
    摘要在那些对象上，造新的就把它丢了。

    `file_path` 写进**段**里而不是留在 chunk 上：跨文件合并后一个 chunk 的段
    可能来自不同文件，来源只有跟着段走才不会串。
    """
    start, end = spans[id(node)]
    return Segment(
        pref=node.pref,
        text=node.content,
        file_path=file_path,
        start_line=start,
        end_line=end,
        tokens=node.content_token,
        special_content=list(node.special_content),
    )


def _subtree_segs(
    node: MDFileHead, spans: dict[int, tuple[int, int]], file_path: str
) -> list[Segment]:
    """把整棵子树按阅读顺序（前序）展开成片段列表。`content` 为空的节点不产出片段。"""
    segs = [_seg_of_node(node, spans, file_path)] if node.content else []
    for child in node.child_head:
        segs.extend(_subtree_segs(child, spans, file_path))
    return segs


@dataclass
class _CutHelper:
    """当前正在攒的那个 chunk 的累加器。跨节点累积，结算后清空复用。

    只攒 `Segment`，**不拼成字符串** —— 面包屑、行号、原文都留在段上，
    什么时候拼、拼不拼摘要，由调用方在 `Segment.render` / `render_text` 那里决定。

    Attributes:
        segments: 已并入的片段，按阅读顺序。
        tokens: 已并入片段的 token 之和。
    """

    segments : list[Segment] = field(default_factory=list)
    tokens : int = 0

    def add_seg(self, seg: Segment) -> None:
        """并入一个片段。"""
        self.segments.append(seg)
        self.tokens += seg.tokens

    def to_draft(self) -> ChunkDraft:
        """结算成 `ChunkDraft`。`file_path` 和 `special_content` 由它自己从 segments 推。"""
        return ChunkDraft(
            segments=list(self.segments),
            tokens=self.tokens,
        )

    def reset(self) -> None:
        """清空，准备下一个 chunk。"""
        self.segments = []
        self.tokens = 0


def _overlop_start(
    lines: list[str], blocks: list[_Block], cut_at: int, overlop: int
) -> int:
    """从切点往回数 `overlop` 个 token，算出下半段该从哪一行重新开始（行下标）。

    回溯出来的这一段就是**重叠区** —— 它已经随上半段切走了，下半段再留一份，
    跨切点的内容从上下两个 chunk 都能召回。

    回溯的起点要同时满足两件事：

    - **不从特殊块中间断开**：起点落在代码块 / 表格内部时**向前跳过整个块**。
      重叠区宁可没有，也不留在下半段的头部 —— 少一个开围栏的代码块，会让下半段
      的块识别整个错位（` ``` ` 被当成普通文本、闭合围栏被当成新开围栏）。
      （往回退到块首是不行的：块首正好是切出去那段的开头，下半段会还原成全文。）
    - **必须真的推进**：起点退到 0 等于下半段还原成全文，切分会原地踏步。
      这时放弃重叠，下半段直接从切点开始。

    Args:
        lines: 当前（剩余）内容的行列表。
        blocks: `_parse_blocks(lines)` 的结果。
        cut_at: 切点（行下标，左闭右开）。
        overlop: 重叠区目标 token 数。

    Returns:
        下半段的新起始行下标，恒 `<= cut_at`。
    """
    if overlop <= 0 or cut_at <= 0:
        return cut_at
    acc = 0
    i = cut_at
    while i > 0 and acc < overlop:
        acc += _count_token(lines[i - 1])
        i -= 1
    # 起点落在硬边界内部 -> 向前跳过整个块（块的下一行就是块尾 + 1）。
    # `text` 不算，切在一段普通正文中间没有问题
    for blk in blocks:
        if blk.kind in ("code", "table") and blk.start < i <= blk.end:
            i = blk.end + 1
            break
    # 没有可用重叠：退到 0 或抵到切点，都会让下半段不推进或还原成全文
    if i <= 0 or i >= cut_at:
        return cut_at
    return i


def _cut_point(
    lines: list[str], blocks: list[_Block], max_token: int
) -> tuple[int, bool]:
    """按 token 预算找切点（行下标，左闭右开），返回 `(切点, 是否因特殊块回退)`。

    逐行走、逐行累加，所以**普通正文可以被从中间切开** —— `_Block` 里 `text` 是软边界。
    代码块和表格是硬边界：预算点落在它们内部时回退到块首。

    切点只保证 `>= 1`，不保证不超限：第一行自己就超预算，或者某个块从第 0 行开始
    且整块超预算时，退无可退，只能整块拿走 —— 这是 `max_token` 标「软约束」的实际含义。

    Args:
        lines: 行列表。
        blocks: `_parse_blocks(lines)` 的结果。
        max_token: 单段 token 上限。

    Returns:
        `(切点, 是否回退过)`。第二个值用来决定「还留不留重叠」。
    """
    # 逐行累加，口径必须和 `_count_token("\n".join(lines[:i]))` 完全一致 ——
    # 包括行与行之间那个换行符。否则这里算出的预算和 `ChunkDraft.tokens` 会对不上，
    # `tokens` 会systematically 超出 max_token
    cjk = 0
    width = 0
    got_content = False  # 是否已经吃到过非空行
    i = 0
    while i < len(lines):
        line = lines[i]
        cjk += len(_CJK_RE.findall(line))
        width += len(line) + (1 if i else 0)  # 行间的 "\n"
        over = cjk + (width - cjk) // 4 > max_token
        # 超预算就停在上一行 —— 但**只有已经吃到正文才算停**。
        # 否则文档以空行开头时，切点会落到只含空行的位置，切出一段空文本
        if over and got_content:
            break
        if line.strip():
            got_content = True
        i += 1
    for blk in blocks:
        # 只有代码块 / 表格是硬边界；`text` 可以被切在任何一行
        if blk.kind in ("code", "table") and blk.start < i <= blk.end:
            # 退到块首；块首就是第 0 行时退无可退，整块拿走
            return (blk.end + 1, True) if blk.start == 0 else (blk.start, True)
    return i, False


def _take_by_budget(text: str, max_token: int) -> int:
    """从 `text` 开头尽量多取字符，使估算 token 不超过 `max_token`。

    口径和 `_count_token` 一致（CJK 按 1 个、其余按 4 字符 1 个，逐字符累加）。

    Args:
        text: 待切文本。
        max_token: token 上限。

    Returns:
        能取的字符数，`1 <= 返回值 <= len(text)` —— 第一个字符就超预算时也得取一个，
        否则切点不推进。
    """
    cjk = 0
    for i, ch in enumerate(text):
        if _CJK_RE.match(ch):
            cjk += 1
        width = i + 1
        if i and cjk + (width - cjk) // 4 > max_token:
            return i
    return len(text)


def _in_line_cut(line: str, max_token: int) -> int:
    """在一行**内部**找切点（字符偏移，左闭右开）。

    **标点优先**：取预算内最靠后的句末标点之后切开 —— 每片尽量接近上限，
    又不把句子劈断。可用的标点一个都没有时**直接按字符硬切**。

    Args:
        line: 待切的那一行。
        max_token: 这一行能用的 token 预算。

    Returns:
        切点字符偏移，恒 `1 <= 返回值 < len(line)`（调用前必须先确认这一行本身超预算）。
    """
    limit = _take_by_budget(line, max_token)
    best = 0
    for matched in _PUNCT_RE.finditer(line, 0, limit + 1):
        best = matched.end()
    return best if best else limit


def _in_line_back(line: str, cut: int, overlop: int) -> int:
    """行内切的重叠起点：从 `cut` 往回数 `overlop` 个 token，返回字符偏移。

    Args:
        line: 被切的那一行。
        cut: 切点（字符偏移）。
        overlop: 重叠区目标 token 数。

    Returns:
        重叠起点的字符偏移；`0` 表示退到头了（调用方此时应当放弃重叠，
        否则下半段会等于原文、切分原地踏步）。
    """
    cjk = 0
    for i in range(cut - 1, -1, -1):
        if _CJK_RE.match(line[i]):
            cjk += 1
        width = cut - i
        if cjk + (width - cjk) // 4 >= overlop:
            return i
    return 0


def content_cut(
    cut_helper:_CutHelper,
    file_head:MDFileHead,
    file_path:str,
    base_line:int,
    base_col:int,
    max_token:int,
    overlop:int|None=None,
) -> tuple[int, int]:
    """把一个节点**自己**的正文切一刀：上半段进 `cut_helper`，下半段原地留下。

    调用前提：`cut_helper` 是空的。

    切点这样选：

    1. 按 token 预算**逐行**推进，正文可以切在任意**行边界**
    2. **一行自己就超过 `max_token`** 时，改在这一行**内部**切：优先落在句末标点之后，
       一个可用标点都没有就按字符硬切（见 `_in_line_cut`）
    3. 代码块和表格是硬边界，切点落进去就回退到块首；整块自己就超预算时整块拿走、
       允许这一段超限，否则切点无法推进（原地死循环）

    下半段会**头部回溯** `overlop` 个 token，形成与上半段的重叠区。
    正文全部装得下时一次拿完、不留剩余，也就不产生重叠。

    Args:
        cut_helper: 累加器，上半段并进这里。
        file_head: 待切的节点。**原地改它的 `content`**，删掉已切走的上半段。
        file_path: 来源文件路径，写进切出来的每个 `Segment`。
        base_line: `file_head.content` 首字符所在行在**清洗后全文**里的行号。
        base_col: 该行内的字符偏移 —— 上一刀切在行内时不为 0。
        max_token: 单段 token 上限（软约束）。
        overlop: 重叠区 token 数；`None` 表示按 `_OVERLOP_RATIO` 从 `max_token` 推。

    Returns:
        下半段的**新起点** `(行号, 行内偏移)`。
    """
    if overlop is None:
        overlop = max(1, int(max_token * _OVERLOP_RATIO))
    content = file_head.content
    lines = content.splitlines()
    blocks = _parse_blocks(lines)
    if not blocks:
        file_head.content = ""
        return base_line, base_col

    # 各行在 content 里的起始偏移（末尾再放一个哨兵），用来在「行下标」和「字符偏移」
    # 之间换算 —— 行内切只有字符偏移表达得了
    line_starts: list[int] = []
    offset = 0
    for item in lines:
        line_starts.append(offset)
        offset += len(item) + 1
    line_starts.append(len(content))

    def _emit(cut_text: str, cut_end: int) -> None:
        """把上半段并进累加器。`special_content` 从节点原来那批对象里筛，
        这样摘要（第 4 步挂在对象上）能跟着进 chunk，不会被重新造的对象丢掉。"""
        cut_helper.add_seg(
            Segment(
                pref=file_head.pref,
                text=cut_text,
                file_path=file_path,
                start_line=base_line,
                end_line=cut_end,
                tokens=_count_token(cut_text),
                special_content=[
                    s
                    for s in file_head.special_content
                    if (s.start_line > base_line or base_col == 0)
                    and base_line <= s.start_line
                    and s.end_line <= cut_end
                ],
            )
        )

    # 1. 第一条非空行；它自己就超过预算时走**行内切**
    first = next((i for i, item in enumerate(lines) if item.strip()), len(lines))
    in_hard_block = any(
        blk.kind in ("code", "table") and blk.start <= first <= blk.end for blk in blocks
    )
    if first < len(lines) and not in_hard_block and _count_token(lines[first]) > max_token:
        line = lines[first]
        cut_col = _in_line_cut(line, max_token)
        cut_offset = line_starts[first] + cut_col
        # 切了 n 个换行就到第 base_line + n 行；这一刀落在行内，所以是 first
        _emit(content[:cut_offset], base_line + first)
        back = _in_line_back(line, cut_col, overlop)
        # back 退到 0 就等于下半段还原成原文，这时放弃重叠，只保证有推进
        new_content = content[cut_offset:] if back <= 0 else content[line_starts[first] + back :]
        file_head.content = new_content
        return (base_line + first, 0) if back <= 0 else (base_line + first, back)

    # 2. 常规路径：按行边界切。落进特殊块就回退，所以不会把代码块 / 表格劈开
    cut_at, pulled_back = _cut_point(lines, blocks, max_token)
    _emit("\n".join(lines[:cut_at]), base_line + cut_at - 1)

    # 3. 全拿完了就没有剩余，也没有重叠可言
    if cut_at >= len(lines):
        file_head.content = ""
        return base_line, base_col

    # 4. 下半段原地留下。
    #    切点是被特殊块顶回来的话**不留重叠**：否则下半段会从重叠处再切一次、又退到
    #    同一个切点，切出一块被上一块完全包住的重复 chunk。
    back = cut_at if pulled_back else _overlop_start(lines, blocks, cut_at, overlop)
    file_head.content = content[line_starts[back] :]
    return base_line + back, 0


def _join_chunks(chunks: list[ChunkDraft]) -> ChunkDraft:
    """把若干 chunk 按阅读顺序拼成一个。

    `ChunkDraft.__post_init__` 会从拼好的 `segments` 推出 `file_path` 与
    `special_content`，所以这里只管拼段、累加 token。`content_hash` 与 `chunk_id`
    要到 `Chunk` 才生成 —— `ChunkDraft` 上没有这两个字段。

    Args:
        chunks: 待拼的 chunk，非空。**可以来自不同文件** —— 来源记在各段上，
            拼完由 `ChunkDraft` 汇总成路径列表。

    Returns:
        拼成的单个 chunk。
    """
    return ChunkDraft(
        segments=[seg for chunk in chunks for seg in chunk.segments],
        tokens=sum(chunk.tokens for chunk in chunks),
    )


def merge_small_chunks(
    chunks: list[ChunkDraft], min_token: int
) -> list[ChunkDraft]:
    """把 token 量小于 `min_token` 的 chunk 并进相邻的 chunk。

    短 chunk 在检索里会当「吸引子」：文本越短，向量越「通用」，跟什么查询都不算远。
    实测（`explore/实际测试/e2e检索测试/干扰块实验.md`）：增量注入 52 个沾边小 chunk
    （24 token）后，**单路向量检索的 MRR@10 从 0.804 掉到 0.372**、Hit@1 从 65.4%
    掉到 15.4%；BM25 侧只掉 0.019，但这批短块被算进平均文档长度，使 fts5 的 avgdl
    从 724.5 降到 609 —— 长文档受到的长度惩罚因此变重。
    并进邻居之后短块只作为上下文存在，不再单独参与召回。

    归宿按优先级逐块扫一遍决定：

    1. **并进上一个** —— 短块多半是紧跟大块之后被 flush 出来的，并回去最自然
    2. **没有上一个就并进下一个** —— 文档开头就是短块时；下一个还没出现，先攒着
    3. **两边都没有就丢弃** —— 整篇只切出一个短块，没有可依附的邻居

    合并**不重新判定**：只认切分产出的原始大小，拼出来的结果再小也不继续找下家。
    只并相邻块，所以合并后的段序仍然连续。

    **跨文件合并也走这里。** `chunk_md_files` 逐文件切成之后把全部结果汇总过来，
    于是「自身不足一个 chunk 的小文件」也能并进相邻文件的块，而不是被丢掉。

    Note:
        合并会让 chunk 超过 `max_token` —— 短块是塞进已经装好的邻居里的，不是重新装箱。
        这是刻意的取舍：短块单独成块对检索的伤害比这一点溢出大。

    Args:
        chunks: 切分产出的 chunk 列表，按阅读顺序。**可以来自不同文件** ——
            来源记在各段上，合并后由 `ChunkDraft` 汇总成路径列表。
        min_token: 判定阈值，`tokens` 严格小于它的算短块。

    Returns:
        合并后的列表。整篇只切出一个短块时返回空列表。
    """
    merged: list[ChunkDraft] = []
    pending: list[ChunkDraft] = []  # 攒下的短块，等下一个正常块出现

    for chunk in chunks:
        if chunk.tokens >= min_token:
            # 正常块：先把它前面攒着的短块并进来（那些短块没有「上一个」）
            merged.append(_join_chunks(pending + [chunk]) if pending else chunk)
            pending = []
        elif merged:
            # 短块，且有上一个：并回去
            merged[-1] = _join_chunks([merged[-1], chunk])
        else:
            # 短块，且没有上一个：攒着等下一个
            pending.append(chunk)

    if pending:
        if merged:
            merged[-1] = _join_chunks([merged[-1], *pending])
        elif len(pending) > 1:
            # 整篇全是短块：合成一个，比全部丢掉强
            merged.append(_join_chunks(pending))
        # 否则整篇只有这一个短块 —— 没有可依附的邻居，丢弃
    return merged


def cut(
    root:MDFileHead,
    max_token:int,
    file_path:str,
    overlop:int|None=None,
)->list[ChunkDraft]:
    """按标题把整棵树切成 chunk。

    切割策略:
        1. 尽量保持同一个节点内容在同一个chunk内 —— 整棵子树装得下就先整块并入
        2. 多个节点如果能够放入一个chunk则放入，若再添加下一个节点无法放入，则放弃下一个节点
        3. 若当前chunk只有一个节点，且正文内容超长需要切割，则需要进行overlop和特殊块保护，
           若当前有多个节点，参考2，放弃该节点进入当前chunk

    Args:
        root: `_build_file_tree` 产出的树（虚拟根），`special_content` 需已填好。
        max_token: 单个 chunk 的 token 上限（软约束，不可切的块会突破）。
        file_path: 写进每个 `Segment.file_path` 的路径串 —— 也就是最终进 `chunk_id`
            的那个值（`chunk_id` 到 `Chunk` 才生成）。传相对路径才能让 ID 跨机器
            可移植；跨文件合并之后，`ChunkDraft.file_path` 是它们的去重列表。
        overlop: 重叠区 token 数，透传给 `content_cut`。

    Returns:
        按阅读顺序排列的 `ChunkDraft` 列表。整篇装得下时只返回一个。

    Note:
        **小 chunk 的合并不在这里做。** 要合并请把结果交给 `merge_small_chunks`
        —— 切分与合并是两个正交的决策；而且**跨文件合并必须先拿到全部文件的结果**，
        放在这里就永远只看得到单个文件。取舍见 `merge_small_chunks`。
    """
    spans = _locate_spans(root)
    chunk_list: list[ChunkDraft] = []
    cut_helper = _CutHelper()

    def flush() -> None:
        """把累加器里的段结算成一个 chunk，并清空。"""
        if not cut_helper.segments:
            return
        chunk_list.append(cut_helper.to_draft())
        cut_helper.reset()

    def _cut(file_head:MDFileHead):
        if file_head is None:
            return

        # 策略 1：整棵子树装得下 -> 优先整块并入当前 chunk
        if 0 < file_head.total_tokens <= max_token:
            if cut_helper.segments and cut_helper.tokens + file_head.total_tokens > max_token:
                flush()
            for seg in _subtree_segs(file_head, spans, file_path):
                cut_helper.add_seg(seg)
            return

        # 装不下 -> 先放本节点正文，再按顺序下钻子节点
        if file_head.content:
            if file_head.content_token <= max_token and (
                not cut_helper.segments
                or cut_helper.tokens + file_head.content_token <= max_token
            ):
                cut_helper.add_seg(_seg_of_node(file_head, spans, file_path))
            else:
                # 策略 3：正文自己就超限 -> 切开，循环到切完为止
                flush()
                base_line = spans[id(file_head)][0]
                base_col = 0
                while file_head.content:
                    base_line, base_col = content_cut(
                        cut_helper,
                        file_head,
                        file_path,
                        base_line,
                        base_col,
                        max_token,
                        overlop,
                    )
                    flush()

        for child in file_head.child_head:
            _cut(child)

    _cut(root)
    flush()
    return chunk_list


def chunk_by_title(
    file_path:Path,
    max_token:int,
    start_line:int,
    end_line:int|None=None,
    summary_func:Callable[[str], str]|None = None,
    source_name:str|None = None)->list[ChunkDraft]:
    """按标题把一个 Markdown 文件切成 chunk 草稿。

    流程：读取文件 -> 按行切片 -> 清洗 -> 建标题树 -> 并发总结特殊块 -> 切分。

    Args:
        file_path: Markdown 文件路径，按 UTF-8 读取。
        max_token: 单个 chunk 的 token 上限（软约束，不可切的块会突破）。
        start_line: 起始行（0-based，左闭）。
        end_line: 结束行（0-based，**右开**）；`None` 表示读到文件末尾。
        summary_func: 总结函数，入参是代码块 / 表格的原文，返回摘要文本。
            **由调用方注入**，本模块不绑定任何模型服务。传 `None` 则整步跳过，
            各块的 `summary` 保持 `None`。单个块抛异常只记 warning，不影响其余块。
        source_name: 写进 `Segment.file_path` 的值 —— 也就是进 `chunk_id` 的那个
            路径。`None`（默认）表示用 `str(file_path)`。要「用绝对路径读文件、
            但让 ID 记相对路径」时传它：跨机器可移植靠的是 **ID 里那个值**，
            与拿什么路径去读无关。
    Returns:
        按阅读顺序排列的 `ChunkDraft` 列表。

    Note:
        **本函数只负责切割，不做小 chunk 合并。** 要合并请把结果交给
        `merge_small_chunks`，或者直接用 `chunk_md_files` 处理整个文件夹
        （它在跨文件汇总之后合并）。

        第 4 步之后的行号是相对**清洗后**文本的；叠加这里的切片偏移后，它已经
        不等于原文件行号。要精确回溯原文需要额外维护映射 —— `_clean_file_content`
        会吃掉连续空行，所以映射不是简单加一个偏移量。

        **`max_token` 是软约束**，唯一会突破的情况是**原子块**（代码块 / 表格）
        自己就超预算 —— 整块保留，不从中间切开。纯文本不会超限：默认切点在行边界，
        一行自己超预算时会改在行内切（标点优先，无标点则按字符硬切）。
        实测 `data/` 全部语料 @1024：477 个 chunk，2 个超限，都是原子块。

        **`max_token` 不含特殊块摘要。** 摘要和面包屑一样是独立字段，不插进
        `Segment.text`，只在算 embedding 时才由 `render_text(..., with_summary=True)`
        拼进去（见 `types.py`）。所以送进 embedding 的文本比 `max_token` 多出
        「本 chunk 内特殊块摘要之和」。设 `max_token` 时请自行留余量。

        之所以不在切分时把摘要算进去：摘要由外部服务生成、失败只 warning，
        一旦它参与 `total_tokens`，同一份文件就会因为一次网络抖动切出不同的结果，
        `chunk_id` 的稳定性随之失效（主键按内容算，而内容不含摘要）。
        若将来实测确有必要，做法是给 `SpecialContent` 加 `token` 字段、总结后回填，
        再重算一次 `total_tokens`（注意那会让切分结果依赖服务可用性）。
    """
    # 1. 读取文件
    content = file_path.read_text(encoding='utf-8')
    content = "\n".join(content.splitlines()[start_line:end_line])
    # 2. 清洗:当前仅仅清洗空格和多行(原样是当前所设计到的文档基本收都是AI或我写的,比较干净)
    # 后续若出现其他文档,再添加别的清洗策略再simple_rag\embedding\clean_pipeline
    clean_content = _clean_file_content(content)
    # 3. 构建文档树
    filetree = _build_file_tree(clean_content)
    # 4. 识别特殊块(表格/代码块)。这一步**总要跑** —— special_content 是事实层,
    #    第 5 步切分要靠它保护特殊块、并把摘要带进 chunk,不只是为了总结
    _fill_special(filetree, _locate_spans(filetree))
    # 4.1 并发总结。先收集待总结的,再并发跑,全部结束后返回;单个失败只 warning。
    #     summary_func 为 None 时跳过(此时 special_content 照样填好,只是没有摘要)
    if summary_func is not None:
        _summarize_specials(_collect_specials(filetree), summary_func)

    # 5. 切分
    return cut(
        filetree, max_token, source_name if source_name is not None else str(file_path)
    )


def chunk_md_files(
    folder_path: Path,
    max_token: int,
    min_token: int,
    start_line: int = 0,
    end_line: int | None = None,
    summary_func: Callable[[str], str] | None = None,
) -> list[ChunkDraft]:
    """切分一个文件夹里的全部 Markdown，并在**跨文件合并**小 chunk 之后返回。

    与「逐文件调 `chunk_by_title`」的区别只有一处，但很关键：**合并发生在汇总之后**。
    于是「自身不足一个 chunk 的小文件」也能并进相邻文件的块，而不是被丢掉；
    单个文件内部找不到邻居的短块，在这里有了跨文件的邻居。

    流程：按路径排序遍历 `**/*.md` → 逐个 `chunk_by_title`（只切割）→ 汇总
    → `merge_small_chunks` 跨文件合并。

    Args:
        folder_path: 文件夹根，**递归**取其中所有 `.md`。
        max_token: 单个 chunk 的 token 上限（软约束），透传给 `chunk_by_title`。
        min_token: 短块阈值（token），透传给 `merge_small_chunks`。**不给默认值** ——
            阈值取多少要按调用方自己的语料实测，这里只给参数、不给建议值。
        start_line: 每个文件的起始行（0-based，左闭）。
        end_line: 每个文件的结束行（0-based，**右开**）；`None` 表示读到文件末尾。
        summary_func: 总结函数，透传给 `chunk_by_title`；`None` 则整步跳过。

    Returns:
        按「文件路径 → 文件内阅读顺序」排列的 `ChunkDraft` 列表。合并只发生在
        相邻块之间，所以整体顺序不变。**整批只有一个短块时返回空列表** ——
        见 `merge_small_chunks` 的归宿规则。

    Note:
        文件按**路径字符串排序**后才逐个处理。顺序不是可有可无的细节：合并的归宿、
        段的先后、`chunk_id` 都由它决定 —— 排序才能让同一批语料每次切出同样的结果。

        路径按 `path.as_posix()` 写进段里（统一用 `/`），顺序也按它排 —— 这样
        Windows 与 POSIX 会得到相同的块序与相同的 `chunk_id`。想要 ID 跨机器可移植，
        `folder_path` 还得是**相对路径**（相对语料根）。
    """
    chunks: list[ChunkDraft] = []
    # 按 **POSIX 形式的路径串**排序，不用 `Path` 默认比较：后者在 Windows 走
    # `normcase`（忽略大小写）、在 POSIX 是大小写敏感，同一批语料在两个平台上
    # 会排出不同的顺序 —— 进而切出不同的 chunk_id。
    for path in sorted(folder_path.rglob("*.md"), key=lambda p: p.as_posix()):
        chunks.extend(
            chunk_by_title(
                path,
                max_token,
                start_line,
                end_line=end_line,
                summary_func=summary_func,
                # 段里也记 POSIX 形式，`\` 与 `/` 的差异不会渗进 chunk_id
                source_name=path.as_posix(),
            )
        )
    return merge_small_chunks(chunks, min_token)



if __name__ =="__main__":
    file_path = Path(r"D:\desktop\软件开发\RAG\data\lifeprismData\diary\2025\01\2025-01-16.md")
    content = file_path.read_text(encoding='utf-8')
    print(content)