"""数据表（`ChunkStore`）的单测：多来源分行、按文件定位、校验、事务。"""

import sqlite3

import pytest

from simple_rag.repository.chunk_store import ChunkStore, Schema


def _store_with(fields=()) -> ChunkStore:
    conn = sqlite3.connect(":memory:")
    store = ChunkStore(conn)
    store.create_table(Schema(fields=fields))
    return store


@pytest.fixture
def store() -> ChunkStore:
    return _store_with()


def _record(chunk_id="c1", content="正文", sources=None, **extra) -> dict:
    """一条记录 = 一个 chunk。默认带一个来源。"""
    if sources is None:
        sources = [{"path": "a.md", "start_line": 1, "end_line": 10}]
    return {"chunk_id": chunk_id, "content": content, "sources": sources, **extra}


# ------------------------------------------------------------------ 建表


def test_create_table可以重复调用(store):
    store.create_table(Schema())  # 第二次不该报 table already exists


def test_建了chunk_id与path两个索引():
    conn = sqlite3.connect(":memory:")
    ChunkStore(conn).create_table(Schema())

    names = {
        row[0]
        for row in conn.execute(
            "select name from sqlite_master "
            "where type = 'index' and tbl_name = 'chunk_data'"
        )
    }
    assert names == {"chunk_data_chunk_id", "chunk_data_path"}


# ------------------------------------------------------------------ 多来源


def test_一个chunk的多个来源分行存放(store):
    store.insert(
        [
            _record(
                sources=[
                    {"path": "a.md", "start_line": 1, "end_line": 10},
                    {"path": "b.md", "start_line": 5, "end_line": 8},
                ]
            )
        ]
    )

    rows = store.get("c1")

    assert [(r.path, r.start_line, r.end_line) for r in rows] == [
        ("a.md", 1, 10),
        ("b.md", 5, 8),
    ]
    assert all(r.content == "正文" for r in rows)  # 正文在多行里重复


def test_get按path与行号排序而不是插入顺序(store):
    store.insert(
        [
            _record(
                sources=[
                    {"path": "z.md", "start_line": 1, "end_line": 2},
                    {"path": "a.md", "start_line": 3, "end_line": 4},
                    {"path": "a.md", "start_line": 1, "end_line": 2},
                ]
            )
        ]
    )

    assert [r.path for r in store.get("c1")] == ["a.md", "a.md", "z.md"]


def test_get不存在的id返回空列表(store):
    assert store.get("nope") == []


def test_同一chunk_id分两次插入来源会并存(store):
    """表上没有唯一约束，重复写入不报错。

    这是刻意留的空白：合并进来的来源、后续补写的来源，都会这样追加。
    调用方要保证幂等，得自己先按 `chunk_id` 清干净再写。
    """
    store.insert([_record(sources=[{"path": "a.md", "start_line": 1, "end_line": 2}])])
    store.insert([_record(sources=[{"path": "b.md", "start_line": 3, "end_line": 4}])])

    assert [r.path for r in store.get("c1")] == ["a.md", "b.md"]


def test_时间戳由模块填且两列同值(store):
    store.insert([_record()])

    row = store.get("c1")[0]

    assert row.created_at == row.updated_at
    assert row.created_at  # 非空，格式 ISO 8601 + UTC


# ------------------------------------------------------------------ 扩展列


def test_扩展列随记录存下来():
    store = _store_with(fields=("tag",))
    store.insert([_record(tag="日记")])

    assert store.get("c1")[0].fields == {"tag": "日记"}


def test_扩展列可以是None():
    store = _store_with(fields=("tag",))
    store.insert([_record(tag=None)])

    assert store.get("c1")[0].fields == {"tag": None}


def test_缺扩展列报错():
    store = _store_with(fields=("tag",))

    with pytest.raises(ValueError, match="缺已声明的字段"):
        store.insert([_record()])


# ------------------------------------------------------------------ 按文件定位


def test_ids_by_path会把同一文件的多个块去重(store):
    store.insert(
        [
            _record(
                chunk_id="c1",
                sources=[
                    {"path": "a.md", "start_line": 1, "end_line": 2},
                    {"path": "a.md", "start_line": 5, "end_line": 6},
                ],
            ),
            _record(
                chunk_id="c2",
                sources=[{"path": "a.md", "start_line": 3, "end_line": 4}],
            ),
        ]
    )

    assert sorted(store.ids_by_path("a.md")) == ["c1", "c2"]


def test_delete_by_path只删该文件的行(store):
    store.insert(
        [
            _record(
                chunk_id="c1",
                sources=[
                    {"path": "a.md", "start_line": 1, "end_line": 2},
                    {"path": "b.md", "start_line": 3, "end_line": 4},
                ],
            ),
            _record(
                chunk_id="c2",
                sources=[{"path": "a.md", "start_line": 5, "end_line": 6}],
            ),
        ]
    )

    assert store.delete_by_path("a.md") == 2
    # c1 还有 b.md 那一行，留着
    assert [(r.chunk_id, r.path) for r in store.get("c1")] == [("c1", "b.md")]
    # c2 只有 a.md 一个来源，连同它的行一起消失
    assert store.get("c2") == []


def test_delete按id删掉全部行(store):
    store.insert(
        [
            _record(
                sources=[
                    {"path": "a.md", "start_line": 1, "end_line": 2},
                    {"path": "b.md", "start_line": 3, "end_line": 4},
                ]
            )
        ]
    )

    assert store.delete("c1") == 2
    assert store.get("c1") == []


def test_delete不存在的id返回0(store):
    assert store.delete("nope") == 0


# ------------------------------------------------------------------ 校验


def test_不认识的键报错(store):
    with pytest.raises(ValueError, match="不认识的键"):
        store.insert([{**_record(), "extra": "x"}])


def test_sources为空报错(store):
    with pytest.raises(ValueError, match="sources 为空"):
        store.insert([_record(sources=[])])


def test_path不能为空(store):
    with pytest.raises(ValueError, match="path 不能为空"):
        store.insert([_record(sources=[{"path": "", "start_line": 1, "end_line": 2}])])


def test_来源里有不认识的键报错(store):
    with pytest.raises(ValueError, match="来源里有不认识的键"):
        store.insert(
            [
                _record(
                    sources=[
                        {
                            "path": "a.md",
                            "start_line": 1,
                            "end_line": 2,
                            "extra": "x",
                        }
                    ]
                )
            ]
        )


def test_start_line不能大于end_line(store):
    with pytest.raises(ValueError, match="start_line 不能大于 end_line"):
        store.insert(
            [_record(sources=[{"path": "a.md", "start_line": 9, "end_line": 2}])]
        )


def test_chunk_id必须是str(store):
    with pytest.raises(TypeError, match="chunk_id 必须是 str"):
        store.insert([_record(chunk_id=123)])


def test_content必须是str(store):
    with pytest.raises(TypeError, match="content 必须是 str"):
        store.insert([_record(content=123)])


def test_行号必须是int(store):
    with pytest.raises(TypeError, match="start_line 必须是 int"):
        store.insert(
            [_record(sources=[{"path": "a.md", "start_line": "1", "end_line": 2}])]
        )


def test_行号不接受bool(store):
    """bool 是 int 的子类，`True` 混进来会静默存成 1。"""
    with pytest.raises(TypeError, match="start_line 必须是 int"):
        store.insert(
            [_record(sources=[{"path": "a.md", "start_line": True, "end_line": 2}])]
        )


# ------------------------------------------------------------------ 事务


def test_一批里有一条非法就整批不落库(store):
    with pytest.raises(ValueError):
        store.insert([_record(chunk_id="c1"), _record(chunk_id="c2", sources=[])])

    assert store.get("c1") == []


def test_没调create_table就操作报错():
    store = ChunkStore(sqlite3.connect(":memory:"))

    with pytest.raises(ValueError, match="还没调 create_table"):
        store.insert([_record()])


def test_插入可以嵌进外层事务():
    conn = sqlite3.connect(":memory:")
    store = ChunkStore(conn)
    store.create_table(Schema())

    conn.execute("savepoint outer")
    store.insert([_record()])
    conn.execute("rollback to outer")
    conn.execute("release outer")

    assert store.get("c1") == []  # 外层回滚把内层一起带走了


# ------------------------------------------------------------------ Schema


def test_扩展列不能占用基本字段名():
    with pytest.raises(ValueError, match="保留名"):
        Schema(fields=("path",))


def test_扩展列名重复大小写不敏感():
    with pytest.raises(ValueError, match="字段名重复"):
        Schema(fields=("tag", "TAG"))


def test_扩展列名不合法():
    with pytest.raises(ValueError, match="不合法"):
        Schema(fields=("tag-name",))


def test_扩展列不能叫sources():
    """`sources` 是记录级的保留键（不是表上的列），撞名会让报错指向错误的原因。"""
    with pytest.raises(ValueError, match="保留名"):
        Schema(fields=("sources",))


def test_SQL关键字可以当扩展列名():
    """DDL 里列名一律加方括号，所以关键字不会撞语法。"""
    store = _store_with(fields=("from",))
    record = _record()
    record["from"] = "x"

    store.insert([record])

    assert store.get("c1")[0].fields == {"from": "x"}


def test_表结构与Schema对不上时整批不落库且事务不挂住():
    """`create_table` 不比对结构（普通表允许 `ALTER`），对不上时插入报
    `no such column` —— 那一步 savepoint 已经开了，必须回滚干净。"""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "create table chunk_data (chunk_id text, path text, start_line integer, "
        "end_line integer, created_at text, updated_at text, content text)"
    )
    store = ChunkStore(conn)
    store.create_table(Schema(fields=("tag",)))  # 表已存在，不会真的加列

    with pytest.raises(sqlite3.OperationalError):
        store.insert([_record(tag="x")])

    assert conn.execute("select count(*) from chunk_data").fetchone()[0] == 0
    assert conn.in_transaction is False  # savepoint 出栈干净，事务没挂着
