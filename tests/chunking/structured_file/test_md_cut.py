"""第 5 步「切分」的单测：整体装填、正文超限切段、重叠、特殊块保护。"""

from pathlib import Path

from simple_rag.chunking.structured_file.types import ChunkDraft, Segment, render_text
from simple_rag.chunking.structured_file.md_chunk_by_title import (
    _build_file_tree,
    _clean_file_content,
    _fill_special,
    _locate_spans,
    chunk_by_title,
    chunk_md_files,
    cut,
    merge_small_chunks,
)


def _write(tmp_path: Path, md: str) -> Path:
    path = tmp_path / "doc.md"
    path.write_text(md, encoding="utf-8")
    return path


def _text(draft) -> str:
    """chunk 的完整文本（面包屑 + 正文），检查内容用。"""
    return render_text(draft.segments)


def _start(draft) -> int:
    """chunk 的起始行 —— chunk 级已不再存这对值，这里取首段的起点。

    这些用例都是单文件的，所以「首段起点 / 末段终点」仍是有效的行区间。
    """
    return draft.segments[0].start_line


def _end(draft) -> int:
    """chunk 的结束行 —— 取末段的终点。"""
    return draft.segments[-1].end_line


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
    assert _start(drafts[0]) == 0


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
    assert _start(drafts[0]) == 0
    assert _end(drafts[-1]) == total_lines - 1
    for before, after in zip(drafts, drafts[1:]):
        assert _end(after) >= _end(before)


def test_重叠_下半段会回溯进上半段的范围(tmp_path):
    md = f"# 标题\n\n{_paras(30)}\n"
    drafts = chunk_by_title(_write(tmp_path, md), max_token=60, start_line=0)

    assert len(drafts) > 1
    first, second = drafts[0], drafts[1]
    assert _start(second) < _end(first)  # 回溯 -> 行号区间重叠
    assert _start(second) > _start(first)  # 但没有回溯到节点开头


def test_重叠_重叠区的每一行在上下两块里都出现(tmp_path):
    md = f"# 标题\n\n{_paras(30)}\n"
    drafts = chunk_by_title(_write(tmp_path, md), max_token=60, start_line=0)

    first, second = drafts[0], drafts[1]
    # 本文档没有连续空行，清洗不会挪行号，所以 md 的行号 == chunk 的行号
    lines = md.splitlines()
    overlap = lines[_start(second) : _end(first) + 1]
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
        assert _start(after) == _end(before) + 1


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
    starts = [_start(d) for d in drafts]
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
    same_line = [d for d in drafts if _start(d) == _end(d) == 2]
    assert len(same_line) > 1
    # 内容不丢（有重叠，所以是 >=）
    assert "".join(_text(d) for d in drafts).count("甲") >= 500


def test_行内切_优先落在句末标点之后(tmp_path):
    sentence = "这是第一句话，内容比较长一些。这是第二句话，也比较长。"
    md = f"# 标题\n\n{sentence * 8}\n"
    drafts = chunk_by_title(_write(tmp_path, md), max_token=100, start_line=0)

    body = [d for d in drafts if _start(d) == 2]
    assert len(body) > 1
    for draft in body[:-1]:  # 最后一段是收尾，不要求标点
        assert _text(draft).rstrip().endswith(("。", "！", "？", "；", "!", "?", ";"))


def test_行内切_没有标点就按字符硬切(tmp_path):
    md = f"# 标题\n\n{'x' * 600}\n"
    drafts = chunk_by_title(_write(tmp_path, md), max_token=60, start_line=0)

    body = [d for d in drafts if _start(d) == 2]
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
    assert _start(drafts[0]) == 0  # 前导空行跟着第一段走，没有被单独扔出来
    assert "".join(_text(d) for d in drafts).count("甲") >= 500


def test_大文档不会死循环且chunk不重复(tmp_path):
    md = f"# 标题\n\n{_paras(200)}\n"
    drafts = chunk_by_title(_write(tmp_path, md), max_token=60, start_line=0)

    assert len(drafts) < 200  # 有重叠但不会退化成「一行一个 chunk」
    ids = [_start(d) for d in drafts]
    assert len(ids) == len(set(ids))  # 起始行互不相同 -> chunk_id 不会撞


# --------------------------------------------------------------- 短块合并
#
# 短 chunk 在检索里会当「吸引子」：文本越短，向量越「通用」，跟什么查询都不算远。
# 实测一个 13 token 的块在多个不相关查询里排到第 1。`min_token` 把这些短块并进邻居，
# 让它们只作为上下文存在，不再单独参与召回。


def _draft(tokens: int, start_line: int, file_path: str = "doc.md") -> ChunkDraft:
    """造一个只含一段的 chunk，token 数可控 —— 单独测合并逻辑用。"""
    seg = Segment(
        pref="标题",
        text="正文",
        file_path=file_path,
        start_line=start_line,
        end_line=start_line,
        tokens=tokens,
    )
    return ChunkDraft(segments=[seg], tokens=tokens)


def _shape(chunks) -> list[int]:
    """各 chunk 的 token 数，用来看合并结果。"""
    return [c.tokens for c in chunks]


def test_短块并进上一个():
    assert _shape(merge_small_chunks([_draft(100, 0), _draft(5, 1)], 30)) == [105]


def test_短块没有上一个就并进下一个():
    assert _shape(merge_small_chunks([_draft(5, 0), _draft(100, 1)], 30)) == [105]


def test_短块两边都没有就丢弃():
    assert merge_small_chunks([_draft(5, 0)], 30) == []


def test_连续短块都并进上一个():
    chunks = [_draft(100, 0), _draft(5, 1), _draft(6, 2)]
    assert _shape(merge_small_chunks(chunks, 30)) == [111]


def test_开头的连续短块并进下一个():
    chunks = [_draft(5, 0), _draft(6, 1), _draft(100, 2)]
    assert _shape(merge_small_chunks(chunks, 30)) == [111]


def test_整篇只有短块时合成一个而不是全丢():
    assert _shape(merge_small_chunks([_draft(5, 0), _draft(6, 1)], 30)) == [11]


def test_没有短块时原样返回():
    chunks = [_draft(100, 0), _draft(200, 1)]
    assert merge_small_chunks(chunks, 30) == chunks


def test_合并把段拼起来而不是丢掉():
    merged = merge_small_chunks([_draft(100, 0), _draft(5, 1)], 30)

    assert len(merged) == 1
    assert len(merged[0].segments) == 2


def test_合并后行区间连续():
    merged = merge_small_chunks([_draft(5, 0), _draft(100, 1), _draft(5, 2)], 30)

    assert _shape(merged) == [110]  # 末尾那个短块并进上一个
    assert (_start(merged[0]), _end(merged[0])) == (0, 2)


def test_切分本身不做合并(tmp_path):
    """`chunk_by_title` 只负责切割：短块照样单独成 chunk，合并在外面显式调用。"""
    md = f"# 标题\n\n{_paras(30)}\n\n# 尾\n\n短。\n"
    drafts = chunk_by_title(_write(tmp_path, md), max_token=60, start_line=0)

    assert min(d.tokens for d in drafts) < 30


def test_合并后短块被并进邻居(tmp_path):
    md = f"# 标题\n\n{_paras(30)}\n\n# 尾\n\n短。\n"
    drafts = chunk_by_title(_write(tmp_path, md), max_token=60, start_line=0)

    assert all(d.tokens >= 30 for d in merge_small_chunks(drafts, 30))


def test_开头的短块并进下一个chunk(tmp_path):
    """文档以一小段正文开头、后面跟超长正文时，开头那段会被单独 flush 出来。"""
    md = "前导一小段。\n\n# 标题\n\n" + "甲" * 300 + "\n"
    drafts = chunk_by_title(_write(tmp_path, md), max_token=60, start_line=0)
    merged = merge_small_chunks(drafts, 30)

    assert all(d.tokens >= 30 for d in merged)
    assert "前导一小段" in _text(merged[0])
    assert "甲" in _text(merged[0])


def test_整篇只有一个短块时返回空列表(tmp_path):
    drafts = chunk_by_title(_write(tmp_path, "# 标题\n"), max_token=1000, start_line=0)

    assert merge_small_chunks(drafts, 30) == []


# ------------------------------------------------------- 跨文件合并（chunk_md_files）


def test_跨文件合并把小文件并进邻居(tmp_path):
    """`chunk_md_files` 存在的理由：小文件自己凑不满一块，跨文件才有邻居。"""
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.md").write_text("# A\n\n短。\n", encoding="utf-8")
    (folder / "b.md").write_text("# B\n\n" + "乙" * 300 + "\n", encoding="utf-8")

    drafts = chunk_md_files(folder, max_token=400, min_token=30)

    assert len(drafts) == 1  # a 并进了 b
    assert drafts[0].file_path == [
        (folder / "a.md").as_posix(),
        (folder / "b.md").as_posix(),
    ]
    assert "短。" in _text(drafts[0])
    assert "乙" in _text(drafts[0])


def test_跨文件合并不改变单一来源(tmp_path):
    """没被合并的块，来源仍是它自己那一个文件。"""
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.md").write_text("# A\n\n" + "甲" * 300 + "\n", encoding="utf-8")

    drafts = chunk_md_files(folder, max_token=60, min_token=30)

    assert drafts
    assert all(d.file_path == [(folder / "a.md").as_posix()] for d in drafts)


def test_段的来源不因合并而串(tmp_path):
    """合并进来的段带着**自己**的路径，而不是整块的第一个路径。"""
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.md").write_text("# A\n\n短。\n", encoding="utf-8")
    (folder / "b.md").write_text("# B\n\n" + "乙" * 300 + "\n", encoding="utf-8")

    chunk = chunk_md_files(folder, max_token=400, min_token=30)[0]

    a_segs = [s for s in chunk.segments if s.file_path == (folder / "a.md").as_posix()]
    b_segs = [s for s in chunk.segments if s.file_path == (folder / "b.md").as_posix()]
    assert a_segs and b_segs
    assert len(a_segs) + len(b_segs) == len(chunk.segments)


def test_文件夹为空时返回空列表(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()

    assert chunk_md_files(folder, max_token=60, min_token=30) == []


# ------------------------------------------------------- 段里记的路径


def test_source_name决定段里记的路径(tmp_path):
    """读文件用文件路径，写进段里的是另一个串 —— ID 的可移植性靠后者。"""
    path = _write(tmp_path, "# 标题\n\n正文。\n")

    drafts = chunk_by_title(path, max_token=1000, start_line=0, source_name="docs/a.md")

    assert drafts[0].file_path == ["docs/a.md"]
    assert drafts[0].segments[0].file_path == "docs/a.md"


def test_不传source_name时用文件路径(tmp_path):
    path = _write(tmp_path, "# 标题\n\n正文。\n")

    drafts = chunk_by_title(path, max_token=1000, start_line=0)

    assert drafts[0].file_path == [str(path)]


def test_chunk_md_files的路径统一用正斜杠(tmp_path):
    """段里记 POSIX 形式，所以路径分隔符的差异不会渗进 chunk_id。"""
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.md").write_text("# A\n\n" + "甲" * 300 + "\n", encoding="utf-8")

    drafts = chunk_md_files(folder, max_token=400, min_token=30)

    assert drafts[0].file_path == [(folder / "a.md").as_posix()]
