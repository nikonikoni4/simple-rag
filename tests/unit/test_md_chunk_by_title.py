"""`md_chunk_by_title.chunk_by_title` 的切分单测。

## 前提假设

输入文档的格式**都是正确的** —— 标题是合法标题（`#` 后有空白）、代码围栏成对、
表格有表头行和分隔行。所以这里只测切分策略本身，不测「格式错误怎么办」。

## 为什么 max_token 都设得很小

切分逻辑的分支只在「装不下」时才触发。用几十 token 的上限，就能在一屏大小的
文档上逼出全部分支（下钻、装箱、正文切分、特殊块保护），不必造长文。

## 场景清单

| # | 测试 | 输入 | max_token | 期望 |
|---|---|---|---|---|
| 1 | `test_只有标题_装得下时合成一个_chunk` | 3 层嵌套标题 | 10000 | 1 块，行区间 (0,2) |
| 2 | `test_只有标题_上限很小时每个标题各成一块` | 同上 | 4 | 3 块，各占 1 行，面包屑逐层加深 |
| 3 | `test_小正文_同层合并到装不下为止` | 3 个同级标题 + 小正文 | 10 | 2 块：甲+乙 合并，丙 单独 |
| 4 | `test_小正文_上限够大时整篇一个_chunk` | 同上 | 10000 | 1 块，三节全在里面 |
| 5 | `test_大正文_被切成多块且每块不超限` | 1 标题 + 5 段正文（约 145 token） | 60 | ≥3 块，每块 ≤60 且都带面包屑 |
| 6 | `test_大正文_切出来的块能拼回原文` | 同上 | 60 | 段落一段不少；标题行只出现 1 次 |
| 7 | `test_无标题纯正文_超限时被切分且不带面包屑` | 5 段正文，无标题 | 60 | ≥2 块，每块 ≤60，无面包屑 |
| 8 | `test_无标题纯正文_整篇装得下时只返回一个` | 同上 | 10000 | 1 块，行区间 (0,8) |
| 9 | `test_代码块_小块跟随正文进同一个_chunk` | 标题 + 3 行代码 | 10000 | 1 块，`special_content == [code]` |
| 10 | `test_代码块_整体超限时不被切开` | 标题 + 60 行代码（约 360 token） | 30 | 代码独占 1 块并超限，围栏成对 |
| 11 | `test_代码块_special_content_记录了行区间` | 标题 + 3 行代码 | 10000 | 行区间合法，文本含首尾围栏 |
| 12 | `test_表格_小表跟随正文进同一个_chunk` | 标题 + 2 行表 | 10000 | 1 块，`special_content == [table]` |
| 13 | `test_表格_整体超限时不被切开` | 标题 + 60 行表（约 300 token） | 30 | 表格独占 1 块并超限，表头未与数据分离 |
| 14 | `test_表格_special_content_记录了行区间` | 标题 + 3 行表 | 10000 | 行区间合法，文本以表头行开头 |
| 15 | `test_代码块与表格同时存在_各自完整` | 标题 + 60 行代码 + 60 行表 | 30 | 各自独占 1 块，互不混入，顺序不乱 |

## 三条贯穿全部用例的约定

1. **自身超限的代码块 / 表格允许超限、整体保留** —— 这是策略不是 bug，所以
   `test_*_整体超限时不被切开` 里显式断言 `tokens > max_token`，把策略钉住：
   将来谁改成「超限就硬切」会立刻红灯。
2. **行区间是闭区间**，相对清洗后的文本；块之间允许跳跃（空行不属于任何块），
   所以 `_assert_ordered` 用严格小于判断，而不是要求首尾相接。
3. **「没被切开」的判据是首尾证据同块**，不是块数：代码看围栏数 `== 2` 且
   首尾两行都在；表格看表头 + 分隔行 + 第一行 + 最后一行都在。
   只数块数证明不了没被从中间切断。
"""

from pathlib import Path

from simple_rag.embedding.structed_file.md_chunk_by_title import chunk_by_title

# 一段长度可控的中文正文：29 个 CJK 字符 + 2 个标点，约 29 token
PARA = "这是一段用于测试切分的正文，长度需要超过上限才能触发切分逻辑。"


# ------------------------------------------------------------------ 工具

def _drafts(tmp_path: Path, md: str, max_token: int):
    """把 md 写进临时文件后切分，返回 ChunkDraft 列表。"""
    fp = tmp_path / "doc.md"
    fp.write_text(md, encoding="utf-8")
    return chunk_by_title(fp, max_token, 0)


def _first_line(draft) -> str:
    """chunk 文本的首行。有面包屑时是面包屑，无面包屑时就是正文首行。"""
    return draft.original_content.splitlines()[0]


def _fence() -> str:
    return "`" * 3


def _assert_ordered(drafts) -> None:
    """行区间必须按阅读顺序排列且互不重叠（允许因空行出现跳跃）。"""
    for prev, nxt in zip(drafts, drafts[1:]):
        assert prev.end_line < nxt.start_line


# ------------------------------------------------------------------ 1. 只含标题

ONLY_HEADINGS = "# 第一章\n## 第一节\n### 第一小节\n"


def test_只有标题_装得下时合成一个_chunk(tmp_path):
    """场景：只有 3 层嵌套标题、没有正文，max_token=10000。

    期望：整篇合成 1 块（覆盖全部 3 行），首行是顶层面包屑，
    三级标题行都还在原文里。
    """
    drafts = _drafts(tmp_path, ONLY_HEADINGS, 10_000)

    assert len(drafts) == 1
    assert (drafts[0].start_line, drafts[0].end_line) == (0, 2)
    assert _first_line(drafts[0]) == "第一章"
    assert "## 第一节" in drafts[0].original_content
    assert "### 第一小节" in drafts[0].original_content


def test_只有标题_上限很小时每个标题各成一块(tmp_path):
    """场景：同样的纯标题文档，max_token=4 —— 每个标题自己就超限。

    期望：下钻到每个标题，各成 1 块、各占 1 行；
    面包屑逐层加深，说明父子关系没被打乱。
    """
    drafts = _drafts(tmp_path, ONLY_HEADINGS, 4)

    assert [_first_line(d) for d in drafts] == [
        "第一章",
        "第一章 > 第一节",
        "第一章 > 第一节 > 第一小节",
    ]
    assert [(d.start_line, d.end_line) for d in drafts] == [(0, 0), (1, 1), (2, 2)]
    _assert_ordered(drafts)


# ------------------------------------------------------------------ 2. 标题 + 小正文

SMALL_BODIES = "# 甲\n甲正文。\n\n# 乙\n乙正文。\n\n# 丙\n丙正文。\n"


def test_小正文_同层合并到装不下为止(tmp_path):
    """场景：3 个同级标题、各带一句小正文，max_token=10 —— 装得下 2 个。

    期望：甲+乙 合并成 1 块，丙 另起 1 块（而不是每节各成一块，
    也不是整篇一块）。
    """
    drafts = _drafts(tmp_path, SMALL_BODIES, 10)

    assert len(drafts) == 2
    assert "甲正文" in drafts[0].original_content
    assert "乙正文" in drafts[0].original_content
    assert "丙正文" not in drafts[0].original_content
    assert "丙正文" in drafts[1].original_content
    assert [(d.start_line, d.end_line) for d in drafts] == [(0, 5), (6, 7)]
    _assert_ordered(drafts)


def test_小正文_上限够大时整篇一个_chunk(tmp_path):
    """场景：同一份小文档，max_token=10000。

    期望：整篇 1 块 —— 上限足够时不产生多余切分。
    """
    drafts = _drafts(tmp_path, SMALL_BODIES, 10_000)

    assert len(drafts) == 1
    for text in ("甲正文", "乙正文", "丙正文"):
        assert text in drafts[0].original_content


# ------------------------------------------------------------------ 3. 正文超过上限

BIG_BODY = "# 章节\n" + "\n\n".join(f"{PARA}{i}" for i in range(1, 6)) + "\n"


def test_大正文_被切成多块且每块不超限(tmp_path):
    """场景：单个标题下 5 段正文（约 145 token），max_token=60 ——
    标题子树装不下，但叶子节点的正文自己就超限。

    期望：走「正文切分」分支切成 ≥3 块；每块都不超限（正文是软边界，
    可以切）；每块都被注入该节点的面包屑。
    """
    max_token = 60
    drafts = _drafts(tmp_path, BIG_BODY, max_token)

    assert len(drafts) >= 3
    for draft in drafts:
        assert draft.tokens <= max_token
        assert _first_line(draft) == "章节"
    _assert_ordered(drafts)


def test_大正文_切出来的块能拼回原文(tmp_path):
    """场景：同上，检查切分是否丢内容。

    期望：5 段正文一段不少；面包屑按块注入（每块首行都是它），
    但标题行 `# 章节` 本身只出现 1 次 —— 它只属于真正含它的那一块，
    不会被复制到每一块。
    """
    max_token = 60
    drafts = _drafts(tmp_path, BIG_BODY, max_token)

    joined = "\n".join(d.original_content for d in drafts)
    for i in range(1, 6):
        assert f"{PARA}{i}" in joined
    assert all(_first_line(d) == "章节" for d in drafts)
    assert joined.count("# 章节") == 1


# ------------------------------------------------------------------ 4. 没有标题

NO_HEADING = "\n\n".join(f"{PARA}{i}" for i in range(1, 6)) + "\n"


def test_无标题纯正文_超限时被切分且不带面包屑(tmp_path):
    """场景：整篇没有任何标题，5 段正文（约 145 token），max_token=60。

    期望：全部内容挂在虚拟根上，正文超限被切成 ≥2 块；每块 ≤60；
    因为虚拟根的 pref 为空，所以**首行就是正文**，不会凭空多出面包屑行。
    """
    max_token = 60
    drafts = _drafts(tmp_path, NO_HEADING, max_token)

    assert len(drafts) >= 2
    for draft in drafts:
        assert draft.tokens <= max_token
        assert _first_line(draft).startswith("这是一段")
    _assert_ordered(drafts)


def test_无标题纯正文_整篇装得下时只返回一个(tmp_path):
    """场景：同一份无标题文档，max_token=10000。

    期望：1 块，覆盖 0~8 行（5 段 + 4 个空行），不因为「没有标题」而报错
    或产出空结果。
    """
    drafts = _drafts(tmp_path, NO_HEADING, 10_000)

    assert len(drafts) == 1
    assert (drafts[0].start_line, drafts[0].end_line) == (0, 8)


# ------------------------------------------------------------------ 5. 代码块

def _code_block(rows: int) -> str:
    body = "\n".join(f"    value_{i} = compute({i})" for i in range(rows))
    return f"{_fence()}python\n{body}\n{_fence()}"


def test_代码块_小块跟随正文进同一个_chunk(tmp_path):
    """场景：标题 + 说明 + 3 行代码块，max_token=10000 —— 代码块很小。

    期望：不因为「它是代码」就强行独立成块，代码跟随正文进同一块；
    但它的类型和行区间被记录在 `special_content` 里（事实层）。
    """
    md = f"# 章节\n说明文字。\n\n{_code_block(3)}\n"

    drafts = _drafts(tmp_path, md, 10_000)

    assert len(drafts) == 1
    assert [s.content_type for s in drafts[0].special_content] == ["code"]
    assert drafts[0].original_content.count(_fence()) == 2


def test_代码块_整体超限时不被切开(tmp_path):
    """场景：标题 + 60 行代码块（约 360 token），max_token=30 ——
    代码块自己就远超上限。

    期望：按策略**允许这一块超限**但整体保留 —— 块数（只含代码的那块恰好
    1 个）不足以证明没被切，所以断言首尾证据：围栏成对、第一行和最后一行
    都在同一块里。
    """
    max_token = 30
    md = f"# 章节\n说明文字。\n\n{_code_block(60)}\n"

    drafts = _drafts(tmp_path, md, max_token)

    holders = [d for d in drafts if any(s.content_type == "code" for s in d.special_content)]
    assert len(holders) == 1, "代码块必须完整落在同一个 chunk 里"

    block = holders[0]
    assert block.tokens > max_token, "自身超限，按策略允许它超限"
    assert block.original_content.count(_fence()) == 2, "首尾围栏都在，没被切断"
    assert "value_0 = compute(0)" in block.original_content
    assert "value_59 = compute(59)" in block.original_content


def test_代码块_special_content_记录了行区间(tmp_path):
    """场景：标题 + 3 行代码块，max_token=10000。

    期望：`special_content` 里的块是**多行**闭区间（起止不相等），
    且文本以围栏开头、以围栏结尾 —— 供第 5 步按区间取摘要用。
    """
    md = f"# 章节\n说明文字。\n\n{_code_block(3)}\n"

    drafts = _drafts(tmp_path, md, 10_000)
    special = drafts[0].special_content[0]

    assert special.start_line < special.end_line
    assert special.content.startswith(_fence())
    assert special.content.endswith(_fence())


# ------------------------------------------------------------------ 6. 表格

def _table(rows: int) -> str:
    body = "\n".join(f"| 项目{i} | {i * 10} |" for i in range(rows))
    return f"| 名称 | 价格 |\n| --- | --- |\n{body}"


def test_表格_小表跟随正文进同一个_chunk(tmp_path):
    """场景：标题 + 说明 + 2 行表，max_token=10000 —— 表格很小。

    期望：表格跟随正文进同一块，类型记为 `table`，表头行还在文本里。
    """
    md = f"# 章节\n说明文字。\n\n{_table(2)}\n"

    drafts = _drafts(tmp_path, md, 10_000)

    assert len(drafts) == 1
    assert [s.content_type for s in drafts[0].special_content] == ["table"]
    assert "| 名称 | 价格 |" in drafts[0].original_content


def test_表格_整体超限时不被切开(tmp_path):
    """场景：标题 + 60 行表格（约 300 token），max_token=30 —— 表格自己就超限。

    期望：允许超限但整体保留。关键断言是**表头没和数据分离** ——
    表头、分隔行、第一行、最后一行四件证据必须同块。少任何一件，
    检索时那批行就是没有列名的裸数字。
    """
    max_token = 30
    md = f"# 章节\n说明文字。\n\n{_table(60)}\n"

    drafts = _drafts(tmp_path, md, max_token)

    holders = [d for d in drafts if any(s.content_type == "table" for s in d.special_content)]
    assert len(holders) == 1, "表格必须完整落在同一个 chunk 里"

    block = holders[0]
    assert block.tokens > max_token, "自身超限，按策略允许它超限"
    assert "| 名称 | 价格 |" in block.original_content
    assert "| --- | --- |" in block.original_content
    assert "| 项目0 | 0 |" in block.original_content
    assert "| 项目59 | 590 |" in block.original_content


def test_表格_special_content_记录了行区间(tmp_path):
    """场景：标题 + 3 行表，max_token=10000。

    期望：`special_content` 是合法的多行闭区间，文本首行就是表头 ——
    后续按区间取摘要时不会把表头漏掉。
    """
    md = f"# 章节\n说明文字。\n\n{_table(3)}\n"

    drafts = _drafts(tmp_path, md, 10_000)
    special = drafts[0].special_content[0]

    assert special.start_line < special.end_line
    assert special.content.splitlines()[0] == "| 名称 | 价格 |"


# ------------------------------------------------------------------ 7. 代码块 + 表格 同时出现

def test_代码块与表格同时存在_各自完整(tmp_path):
    """场景：同一个标题下，60 行代码块 紧跟 60 行表格，max_token=30 ——
    两个特殊块都超限。

    期望：各自独占一块且互不混入（不是同一对象）；代码围栏成对；
    表格末行还在；全篇行区间仍按顺序不重叠。
    """
    max_token = 30
    md = f"# 章节\n说明文字。\n\n{_code_block(60)}\n\n{_table(60)}\n"

    drafts = _drafts(tmp_path, md, max_token)

    code_holders = [d for d in drafts if any(s.content_type == "code" for s in d.special_content)]
    table_holders = [d for d in drafts if any(s.content_type == "table" for s in d.special_content)]
    assert len(code_holders) == 1
    assert len(table_holders) == 1
    assert code_holders[0] is not table_holders[0]
    assert code_holders[0].original_content.count(_fence()) == 2
    assert "| 项目59 | 590 |" in table_holders[0].original_content
    _assert_ordered(drafts)
