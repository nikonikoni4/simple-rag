"""第 5 步「切分」的单测：整体装填、正文超限切段、重叠、特殊块保护。"""

from pathlib import Path

from simple_rag.embedding.structed_file.types import render_text
from simple_rag.embedding.structed_file.md_chunk_by_title import (
    _build_file_tree,
    _clean_file_content,
    _fill_special,
    _locate_spans,
    chunk_by_title,
    cut,
)


def _write(tmp_path: Path, md: str) -> Path:
    path = tmp_path / "doc.md"
    path.write_text(md, encoding="utf-8")
    return path


def _text(draft) -> str:
    """chunk 的完整文本（面包屑 + 正文），检查内容用。"""
    return render_text(draft.segments)


def _tree_of(md: str):
    """建树、填好 special_content 并回填 token，供直接调 `cut` 用。"""
    tree = _build_file_tree(_clean_file_content(md))
    _fill_special(tree, _locate_spans(tree))
    return tree


def _paras(n: int) -> str:
    """n 个正文段落，段间空行 —— 空行是 `_parse_blocks` 的软边界。"""
    return "\n\n".join(f"这是第{i}段正文内容。" for i in range(n))


def _code(lines: int) -> str:
    return "```python\n" + "\n".join(f"x{i} = {i}" for i in range(lines)) + "\n```"


def test_整篇装得下时只出一个chunk(tmp_path):
    drafts = chunk_by_title(
        _write(tmp_path, "# 标题\n\n正文。\n\n## 子标题\n\n子正文。\n"),
        max_token=10000,
        start_line=0,
    )

    assert len(drafts) == 1
    assert "正文。" in _text(drafts[0])
    assert "子正文。" in _text(drafts[0])
    assert drafts[0].start_line == 0


def test_多个节点装得下就合并成一个chunk(tmp_path):
    md = f"# A\n\n{'甲' * 50}\n\n# B\n\n{'乙' * 50}\n"
    drafts = chunk_by_title(_write(tmp_path, md), max_token=110, start_line=0)

    assert len(drafts) == 1
    assert "甲" in _text(drafts[0]) and "乙" in _text(drafts[0])


def test_装不下就另起一个chunk(tmp_path):
    md = f"# A\n\n{'甲' * 50}\n\n# B\n\n{'乙' * 50}\n"
    drafts = chunk_by_title(_write(tmp_path, md), max_token=60, start_line=0)

    assert len(drafts) == 2
    assert "甲" in _text(drafts[0]) and "乙" not in _text(drafts[0])
    assert "乙" in _text(drafts[1])


def test_正文超限时切成多段且不超限(tmp_path):
    md = f"# 标题\n\n{_paras(30)}\n"
    drafts = chunk_by_title(_write(tmp_path, md), max_token=60, start_line=0)

    assert len(drafts) > 1
    for draft in drafts:
        assert draft.tokens <= 60


def test_每段都带上自己的面包屑(tmp_path):
    md = f"# 甲章\n\n{_paras(20)}\n\n# 乙章\n\n乙章正文。\n"
    drafts = chunk_by_title(_write(tmp_path, md), max_token=60, start_line=0)

    assert all("甲章" in _text(d) for d in drafts[:-1])
    assert "乙章" in _text(drafts[-1])


def test_chunk按阅读顺序推进到文末(tmp_path):
    md = f"# 标题\n\n{_paras(30)}\n"
    drafts = chunk_by_title(_write(tmp_path, md), max_token=60, start_line=0)

    total_lines = len(md.splitlines())
    assert drafts[0].start_line == 0
    assert drafts[-1].end_line == total_lines - 1
    for before, after in zip(drafts, drafts[1:]):
        assert after.end_line >= before.end_line


def test_重叠_下半段会回溯进上半段的范围(tmp_path):
    md = f"# 标题\n\n{_paras(30)}\n"
    drafts = chunk_by_title(_write(tmp_path, md), max_token=60, start_line=0)

    assert len(drafts) > 1
    first, second = drafts[0], drafts[1]
    assert second.start_line < first.end_line  # 回溯 -> 行号区间重叠
    assert second.start_line > first.start_line  # 但没有回溯到节点开头


def test_重叠_重叠区的每一行在上下两块里都出现(tmp_path):
    md = f"# 标题\n\n{_paras(30)}\n"
    drafts = chunk_by_title(_write(tmp_path, md), max_token=60, start_line=0)

    first, second = drafts[0], drafts[1]
    # 本文档没有连续空行，清洗不会挪行号，所以 md 的行号 == chunk 的行号
    lines = md.splitlines()
    overlap = lines[second.start_line : first.end_line + 1]
    assert overlap, "相邻两块之间应该有重叠区"
    for line in overlap:
        assert line in _text(first)
        assert line in _text(second)


def test_重叠_可以关掉(tmp_path):
    """`cut` 的 overlop 传 0 时不产生重叠，两块首尾相接。"""
    md = f"# 标题\n\n{_paras(30)}\n"
    tree = _tree_of(md)

    drafts = cut(tree, max_token=60, file_path="doc.md", overlop=0)

    assert len(drafts) > 1
    for before, after in zip(drafts, drafts[1:]):
        assert after.start_line == before.end_line + 1


def test_代码块不会被从中间切开(tmp_path):
    md = f"# 标题\n\n前言。\n\n{_code(40)}\n\n后记。\n"
    drafts = chunk_by_title(_write(tmp_path, md), max_token=60, start_line=0)

    for draft in drafts:
        assert _text(draft).count("```") % 2 == 0  # 围栏成对
    assert any("x39 = 39" in _text(d) for d in drafts)


def test_表格不会被从中间切开(tmp_path):
    table = "| 项目 | 值 |\n|---|---|\n" + "\n".join(f"| 行{i} | {i} |" for i in range(30))
    md = f"# 标题\n\n前言。\n\n{table}\n\n后记。\n"
    drafts = chunk_by_title(_write(tmp_path, md), max_token=60, start_line=0)

    holders = [d for d in drafts if "行0 |" in _text(d)]
    assert holders, "表格应该出现在某个 chunk 里"
    assert all("行29 |" in _text(d) for d in holders)  # 整张表都在


def test_切点贴着代码块时不会退化成一行一推进(tmp_path):
    """回归：回溯起点落进代码块后，曾经每次只前进 1 行，切出 20+ 段。"""
    md = f"# 标题\n\n前言。\n\n{_code(40)}\n\n后记。\n"
    drafts = chunk_by_title(_write(tmp_path, md), max_token=60, start_line=0)

    assert len(drafts) <= 5
    starts = [d.start_line for d in drafts]
    assert len(starts) == len(set(starts))


def test_块自己就超限时整块保留允许超限(tmp_path):
    md = f"# 标题\n\n{_code(200)}\n"
    drafts = chunk_by_title(_write(tmp_path, md), max_token=50, start_line=0)

    holder = [d for d in drafts if "x199 = 199" in _text(d)]
    assert len(holder) == 1  # 不丢内容
    assert holder[0].tokens > 50  # 软约束：允许这一段超限
    assert _text(holder[0]).count("```") == 2


def test_不传summary时特殊块仍然被识别并带进chunk(tmp_path):
    md = "# 标题\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    drafts = chunk_by_title(_write(tmp_path, md), max_token=10000, start_line=0)

    specials = [s for d in drafts for s in d.special_content]
    assert [s.content_type for s in specials] == ["table"]
    assert specials[0].summary is None


def test_摘要随特殊块一起进chunk(tmp_path):
    md = "# 标题\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    drafts = chunk_by_title(
        _write(tmp_path, md), max_token=10000, start_line=0, summary_func=lambda text: "表摘要"
    )

    specials = [s for d in drafts for s in d.special_content]
    assert [s.summary for s in specials] == ["表摘要"]


def test_无空行的长正文也能切开(tmp_path):
    """回归：连续非空行会被 `_parse_blocks` 合成一个 `text` 块，但它是软边界。

    修复前「块首在第 0 行 -> 整块拿走」对 text 块也生效，于是整篇正文被一次端走，
    真实语料上切出过 8 倍 max_token 的 chunk。
    """
    md = "# 标题\n\n" + "\n".join(
        f"- 这是第{i}条很长的列表项内容，用来把这一块撑大。" for i in range(40)
    ) + "\n"
    drafts = chunk_by_title(_write(tmp_path, md), max_token=60, start_line=0)

    assert len(drafts) > 1
    for draft in drafts:
        assert draft.tokens <= 60


def test_超长单行会被行内切开(tmp_path):
    """一行自己超过 `max_token` 时改在行内切，不再整行端走。"""
    md = f"# 标题\n\n{'甲' * 500}\n"
    drafts = chunk_by_title(_write(tmp_path, md), max_token=60, start_line=0)

    assert len(drafts) > 1
    assert all(d.tokens <= 60 for d in drafts)
    # 多段来自同一行：行号一样，只靠内容区分（主键也按内容算，所以不会撞）
    same_line = [d for d in drafts if d.start_line == d.end_line == 2]
    assert len(same_line) > 1
    # 内容不丢（有重叠，所以是 >=）
    assert "".join(_text(d) for d in drafts).count("甲") >= 500


def test_行内切_优先落在句末标点之后(tmp_path):
    sentence = "这是第一句话，内容比较长一些。这是第二句话，也比较长。"
    md = f"# 标题\n\n{sentence * 8}\n"
    drafts = chunk_by_title(_write(tmp_path, md), max_token=100, start_line=0)

    body = [d for d in drafts if d.start_line == 2]
    assert len(body) > 1
    for draft in body[:-1]:  # 最后一段是收尾，不要求标点
        assert _text(draft).rstrip().endswith(("。", "！", "？", "；", "!", "?", ";"))


def test_行内切_没有标点就按字符硬切(tmp_path):
    md = f"# 标题\n\n{'x' * 600}\n"
    drafts = chunk_by_title(_write(tmp_path, md), max_token=60, start_line=0)

    body = [d for d in drafts if d.start_line == 2]
    assert len(body) > 1
    assert all(d.tokens <= 60 for d in body)


def test_行内切_不切进代码块里(tmp_path):
    """代码块是硬边界，里面就算有超长行也不许行内切。"""
    md = f"# 标题\n\n```python\n{'y' * 600}\n```\n"
    drafts = chunk_by_title(_write(tmp_path, md), max_token=60, start_line=0)

    holders = [d for d in drafts if "y" * 600 in _text(d)]
    assert len(holders) == 1  # 整块在一个 chunk 里，没被劈开
    assert _text(holders[0]).count("```") == 2
    assert holders[0].tokens > 60  # 允许超限


def test_前导空行不会切出空chunk(tmp_path):
    """回归：文档以空行开头时，切点曾落到只含空行的位置，切出一段空文本。

    旧实现用「渲染后的字符串空不空」判空，把它静默吞了；改成按段累积后漏了出来。
    """
    md = "\n\n" + "甲" * 500 + "\n"
    drafts = chunk_by_title(_write(tmp_path, md), max_token=60, start_line=0)

    assert all(_text(d).strip() for d in drafts)
    assert drafts[0].start_line == 0  # 前导空行跟着第一段走，没有被单独扔出来
    assert "".join(_text(d) for d in drafts).count("甲") >= 500


def test_大文档不会死循环且chunk不重复(tmp_path):
    md = f"# 标题\n\n{_paras(200)}\n"
    drafts = chunk_by_title(_write(tmp_path, md), max_token=60, start_line=0)

    assert len(drafts) < 200  # 有重叠但不会退化成「一行一个 chunk」
    ids = [d.start_line for d in drafts]
    assert len(ids) == len(set(ids))  # 起始行互不相同 -> chunk_id 不会撞
