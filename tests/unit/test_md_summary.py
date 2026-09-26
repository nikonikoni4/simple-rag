"""第 4 步「特殊块识别 + 并发总结」的单测。

只测这一步的内部管线（`_build_file_tree` -> `_fill_special` -> `_collect_specials`
-> `_summarize_specials`）。走 `chunk_by_title` 测不了 —— 树的产物要等第 5 步切分
才会以 `ChunkDraft` 的形式露出来，而切分还没实现。
"""

import logging

from simple_rag.embedding.structed_file.md_chunk_by_title import (
    _build_file_tree,
    _collect_specials,
    _fill_special,
    _locate_spans,
    _summarize_specials,
)

_TABLE = "| 项目 | 值 |\n|---|---|\n| a | 1 |\n| b | 2 |"
_CODE = "```python\nprint('hi')\nprint('ho')\n```"

_MD = f"""# 章节

说明文字。

{_TABLE}

正文段落。

{_CODE}
"""


def _tree(md: str = _MD):
    """建树并填好 `special_content`，返回虚拟根节点。"""
    tree = _build_file_tree(md)
    _fill_special(tree, _locate_spans(tree))
    return tree


def test_收集_表格与代码块都被收集_文本不收集():
    blocks = _collect_specials(_tree())

    assert [b.content_type for b in blocks] == ["table", "code"]
    assert "| 项目 | 值 |" in blocks[0].content
    assert "print('ho')" in blocks[1].content
    assert all(b.summary is None for b in blocks)


def test_收集_按阅读顺序排列():
    blocks = _collect_specials(_tree())

    assert [b.start_line for b in blocks] == sorted(b.start_line for b in blocks)


def test_收集_已有摘要的块不再收集():
    tree = _tree()
    blocks = _collect_specials(tree)
    blocks[0].summary = "已经总结过了"

    # 过滤是看 summary 字段,所以原来那块 table 不再出现
    assert [b.content_type for b in _collect_specials(tree)] == ["code"]


def test_总结_摘要写回各自的块():
    blocks = _collect_specials(_tree())

    _summarize_specials(blocks, lambda text: f"摘要:{text.splitlines()[0]}")

    assert [b.summary for b in blocks] == ["摘要:| 项目 | 值 |", "摘要:```python"]


def test_总结_空列表不报错():
    _summarize_specials([], lambda text: "不会被调用")


def test_总结_单块失败只warning不抛(caplog):
    blocks = _collect_specials(_tree())

    def _flaky(text: str) -> str:
        if "项目" in text:
            raise RuntimeError("模拟服务端 500")
        return "ok"

    with caplog.at_level(logging.WARNING):
        _summarize_specials(blocks, _flaky)  # 不抛

    table_block, code_block = blocks
    assert table_block.summary is None  # 失败的那块保持 None
    assert code_block.summary == "ok"  # 其余块照常拿到摘要
    assert len(caplog.records) == 1
    assert "总结失败" in caplog.records[0].getMessage()
    assert "行" in caplog.records[0].getMessage()  # 定位信息:行号进了日志


def test_总结_全部失败也不抛(caplog):
    blocks = _collect_specials(_tree())

    def _boom(text: str) -> str:
        raise RuntimeError("全挂")

    with caplog.at_level(logging.WARNING):
        _summarize_specials(blocks, _boom)

    assert all(b.summary is None for b in blocks)
    assert len(caplog.records) == 2


def test_总结_并发跑完再返回():
    """`_summarize_specials` 返回时所有块都必须已有结果(即 wait 住了)。"""
    blocks = _collect_specials(_tree())

    _summarize_specials(blocks, lambda text: "done")

    assert all(b.summary == "done" for b in blocks)


def test_总结_结果与完成顺序无关():
    """写回的是各块自己的对象,所以慢块不会串到快块上。"""
    blocks = _collect_specials(_tree())

    def _slow_for_table(text: str) -> str:
        return "table" if "项目" in text else "code"

    _summarize_specials(blocks, _slow_for_table)

    assert [b.summary for b in blocks] == ["table", "code"]
