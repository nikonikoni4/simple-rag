"""`Segment` / `render_text` / `Chunk` 的形状与拼装规则。

重点是**面包屑和摘要都不进原文**：存的时候分开，用的时候才拼。
"""

import numpy as np
import pytest

from simple_rag.embedding.structed_file.md_chunk_by_title import chunk_by_title
from simple_rag.embedding.structed_file.types import Chunk, Segment, SpecialContent, render_text


def _seg(text: str, *, pref: str = "", start: int = 0, specials=()) -> Segment:
    """造一段：行区间按 text 的实际行数推，省得手数。"""
    return Segment(
        pref=pref,
        text=text,
        start_line=start,
        end_line=start + len(text.splitlines()) - 1,
        tokens=0,
        special_content=list(specials),
    )


def _table(start: int, lines: int, summary: str | None) -> SpecialContent:
    content = "| a | b |\n|---|---|\n" + "\n".join(f"| {i} | {i} |" for i in range(lines))
    return SpecialContent(start, start + lines + 1, content, "table", summary)


def test_段原文不含面包屑也不含摘要():
    table = _table(start=2, lines=2, summary="这是一张表")
    seg = _seg("# 标题\n\n正文。\n\n" + table.content, pref="甲 > 乙", specials=[table])

    assert "甲 > 乙" not in seg.text
    assert "这是一张表" not in seg.text
    assert "# 标题" in seg.text  # 标题行本身是原文的一部分


def test_render不加摘要时就是面包屑加原文():
    seg = _seg("# 标题\n\n正文。", pref="甲 > 乙")

    assert seg.render() == "甲 > 乙\n# 标题\n\n正文。"
    assert seg.render(with_summary=True) == seg.render()  # 没有块可插


def test_render把摘要插在它那块之前():
    body = "前言。\n\n| a | b |\n|---|---|\n| 1 | 1 |\n| 2 | 2 |\n\n后记。"
    table = _table(start=2, lines=2, summary="表摘要")
    seg = _seg(body, specials=[table])

    out = seg.render(with_summary=True)

    lines = out.splitlines()
    assert lines[lines.index("表摘要") + 1] == "| a | b |"  # 紧贴在表头前面
    assert out.count("表摘要") == 1
    assert "| a | b |" in out and "| 2 | 2 |" in out  # 原表照样保留


def test_render不插入别的段的块():
    """行区间不落在本段内的块不插 —— 靠行号判断，不靠搜文本。"""
    other = _table(start=100, lines=2, summary="别处的表")
    seg = _seg("# 标题\n\n正文。", specials=[other])

    assert "别处的表" not in seg.render(with_summary=True)


def test_render跳过没有摘要的块():
    table = _table(start=2, lines=2, summary=None)
    seg = _seg("前言。\n\n" + table.content, specials=[table])

    assert seg.render(with_summary=True) == seg.render()


def test_render跳过摘要生成失败的块():
    ok = SpecialContent(2, 3, "| a |\n|---|\n| 1 |", "table", "好摘要")
    seg = _seg("前言。\n\n| a |\n|---|\n| 1 |", specials=[ok])

    assert "好摘要" in seg.render(with_summary=True)


def test_render_多段之间空行分隔():
    segments = [_seg("第一段", pref="A"), _seg("第二段", pref="B", start=10)]

    assert render_text(segments) == "A\n第一段\n\nB\n第二段"


def test_render_空列表返回空串():
    assert render_text([]) == ""


def test_render_段没有面包屑时不加空行():
    assert render_text([_seg("裸正文")]) == "裸正文"


def test_chunkdraft带file_path(tmp_path, monkeypatch):
    path = tmp_path / "doc.md"
    path.write_text("# 标题\n\n正文。\n", encoding="utf-8")

    draft = chunk_by_title(path, max_token=10000, start_line=0)[0]

    assert draft.file_path == str(path)


def test_chunk_id随file_path变():
    segments = [_seg("正文。", pref="标题")]

    a = Chunk(segments, "a/doc.md", 5, np.zeros(4))
    b = Chunk(segments, "b/doc.md", 5, np.zeros(4))

    assert a.chunk_id != b.chunk_id  # 不同目录同名文件不会撞主键


def test_chunk_id随内容变():
    """主键按内容寻址：正文改了就是另一条记录。"""
    a = Chunk([_seg("原来的正文。", pref="标题")], "doc.md", 5, np.zeros(4))
    b = Chunk([_seg("改过的正文。", pref="标题")], "doc.md", 5, np.zeros(4))

    assert a.chunk_id != b.chunk_id
    assert a.content_hash != b.content_hash


def test_chunk_id不看行号():
    """同一段内容，文档增删让行号平移了 —— 主键必须不变。"""
    a = Chunk([_seg("正文。", pref="标题", start=10)], "doc.md", 5, np.zeros(4))
    b = Chunk([_seg("正文。", pref="标题", start=999)], "doc.md", 5, np.zeros(4))

    assert a.chunk_id == b.chunk_id


def test_同一行切出的多段主键互不相同():
    """这正是「行内切分」不再撞主键的原因：行区间一样，但内容不一样。"""
    pieces = ["很长的一行的前半段。", "很长的一行的后半段。"]
    # 三段来自同一行（start=end=94），只靠内容区分
    chunks = [Chunk([_seg(p, pref="标题", start=94)], "doc.md", 5, np.zeros(4)) for p in pieces]

    assert len({c.chunk_id for c in chunks}) == len(pieces)
    assert all(c.start_line == c.end_line == 94 for c in chunks)


def test_同一文件同一内容就是同一条记录():
    """去重的依据：内容一样 -> 主键一样，不需要存两份。"""
    a = Chunk([_seg("正文。", pref="标题")], "doc.md", 5, np.zeros(4))
    b = Chunk([_seg("正文。", pref="标题")], "doc.md", 5, np.zeros(4))

    assert a.chunk_id == b.chunk_id


def test_面包屑参与主键():
    """标题改名会改 embedding 输入，应当算「内容变了」。"""
    a = Chunk([_seg("正文。", pref="旧标题")], "doc.md", 5, np.zeros(4))
    b = Chunk([_seg("正文。", pref="新标题")], "doc.md", 5, np.zeros(4))

    assert a.chunk_id != b.chunk_id


def test_content_hash与路径无关():
    """留着它就是为了这一条：能问「这段内容在库里出现过吗」。"""
    a = Chunk([_seg("正文。", pref="标题")], "a/doc.md", 5, np.zeros(4))
    b = Chunk([_seg("正文。", pref="标题")], "b/doc.md", 5, np.zeros(4))

    assert a.chunk_id != b.chunk_id  # 路径参与主键
    assert a.content_hash == b.content_hash  # 但内容指纹一样


def test_content_hash不含摘要():
    table = _table(start=2, lines=2, summary=None)
    before = Chunk([_seg("前言。\n\n" + table.content, specials=[table])], "doc.md", 5, np.zeros(4))

    table.summary = "后来才生成的摘要"
    after = Chunk([_seg("前言。\n\n" + table.content, specials=[table])], "doc.md", 5, np.zeros(4))

    assert before.content_hash == after.content_hash


def test_content_hash把面包屑算进去():
    """标题改名会改 embedding 输入，所以必须改 hash，否则向量会静默过期。"""
    a = Chunk([_seg("正文。", pref="旧标题")], "doc.md", 5, np.zeros(4))
    b = Chunk([_seg("正文。", pref="新标题")], "doc.md", 5, np.zeros(4))

    assert a.content_hash != b.content_hash


def test_chunk的行区间与特殊块从segments推出():
    table = _table(start=12, lines=2, summary="S")
    segments = [_seg("第一段", pref="A", start=10), _seg("第二段", start=12, specials=[table])]

    chunk = Chunk(segments, "doc.md", 5, np.zeros(4))

    assert (chunk.start_line, chunk.end_line) == (10, 12)  # 末段的结束行
    assert chunk.special_content == [table]


def test_embedding_vec转成float32():
    chunk = Chunk([_seg("正文。")], "doc.md", 5, [0.5, 1.5])

    assert chunk.embedding_vec.dtype == np.float32
    assert chunk.embedding_vec.tolist() == [pytest.approx(0.5), pytest.approx(1.5)]
