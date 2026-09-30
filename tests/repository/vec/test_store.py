"""`_store` 的纯单测 —— 只构造 SQL 文本，不碰数据库。"""

import pytest

from simple_rag.repository.vec import _store
from simple_rag.repository.vec._schema import Schema

SCHEMA = Schema(
    dim=8,
    metric="cosine",
    fields=("loc", "content", "from"),
    filterable=("loc", "from"),
)


# ---------------------------------------------------------------- check_k

@pytest.mark.parametrize("k", [0, 1, 5, 4096])
def test_k_合法(k):
    assert _store.check_k(k) == k


@pytest.mark.parametrize("k", [-1, 4097, 100000])
def test_k_越界(k):
    with pytest.raises(ValueError, match="k 必须在"):
        _store.check_k(k)


@pytest.mark.parametrize("k", [True, False, 5.0, "5", None])
def test_k_类型不对(k):
    with pytest.raises(TypeError, match="k 必须是 int"):
        _store.check_k(k)


# ---------------------------------------------------------------- check_where

def test_where_为_None_就是不筛():
    assert _store.check_where(SCHEMA, None) == []


def test_where_空_dict_就是不筛():
    assert _store.check_where(SCHEMA, {}) == []


def test_where_允许_chunk_id_和可过滤字段():
    assert _store.check_where(SCHEMA, {"loc": "a.md", "chunk_id": "c-1"}) == [
        ("loc", "=", "a.md"),
        ("chunk_id", "=", "c-1"),
    ]


def test_where_str_值归一化成等值():
    assert _store.check_where(SCHEMA, {"loc": "a.md"}) == [("loc", "=", "a.md")]


@pytest.mark.parametrize("op", ["=", ">", ">=", "<", "<="])
def test_where_元组表达比较_五种操作符(op):
    assert _store.check_where(SCHEMA, {"loc": (op, " 2025-06-01")}) == [
        ("loc", op, " 2025-06-01")
    ]


def test_where_元组的操作符不支持就报错():
    with pytest.raises(ValueError, match="操作符"):
        _store.check_where(SCHEMA, {"loc": ("~=", "a")})


def test_where_元组的比较值必须是_str():
    with pytest.raises(TypeError, match="比较值必须是 str"):
        _store.check_where(SCHEMA, {"loc": (">=", 5)})


def test_where_元组长度不对报错():
    with pytest.raises(TypeError, match="必须是 str"):
        _store.check_where(SCHEMA, {"loc": (">=", "a", "b")})


def test_where_列表不算元组():
    with pytest.raises(TypeError, match="必须是 str"):
        _store.check_where(SCHEMA, {"loc": [">=", "a"]})


def test_where_不许过滤加号列():
    """加号列在 KNN 里过滤不了（实测探针 18）—— 模块把它拦在 SQL 之外。"""
    with pytest.raises(ValueError, match="不能用于过滤"):
        _store.check_where(SCHEMA, {"content": "x"})


def test_where_不许过滤未声明的字段():
    with pytest.raises(ValueError, match="不能用于过滤"):
        _store.check_where(SCHEMA, {"nosuchfield": "x"})


@pytest.mark.parametrize("value", [1, None, b"x", ["a"]])
def test_where_的值类型不对(value):
    with pytest.raises(TypeError, match="必须是 str"):
        _store.check_where(SCHEMA, {"loc": value})


def test_where_必须是_mapping():
    with pytest.raises(TypeError, match="必须是 mapping"):
        _store.check_where(SCHEMA, ["loc"])


# ---------------------------------------------------------------- check_record

def good(**kwargs) -> dict:
    base = {"chunk_id": "c-1", "vector": [1, 0, 0, 0, 0, 0, 0, 0], "loc": "a.md",
            "content": "正文", "from": "x"}
    return {**base, **kwargs}


def test_合法记录返回_chunk_id():
    assert _store.check_record(SCHEMA, good()) == "c-1"


def test_字段值可以是_None():
    assert _store.check_record(SCHEMA, good(content=None)) == "c-1"


@pytest.mark.parametrize("missing", ["chunk_id", "vector"])
def test_缺必需键(missing):
    record = good()
    del record[missing]
    with pytest.raises(ValueError, match="缺必需的键"):
        _store.check_record(SCHEMA, record)


@pytest.mark.parametrize("field", ["loc", "content", "from"])
def test_缺已声明字段(field):
    record = good()
    del record[field]
    with pytest.raises(ValueError, match="缺已声明的字段"):
        _store.check_record(SCHEMA, record)


@pytest.mark.parametrize("key", ["created_at", "updated_at", "nosuchfield", "Loc"])
def test_不认识的键不静默忽略(key):
    with pytest.raises(ValueError, match="不认识的键"):
        _store.check_record(SCHEMA, good(**{key: "x"}))


@pytest.mark.parametrize("value", [123, b"x", ["a"]])
def test_字段值类型不对(value):
    with pytest.raises(TypeError, match="必须是 str 或 None"):
        _store.check_record(SCHEMA, good(loc=value))


@pytest.mark.parametrize("value", [123, None, b"c-1"])
def test_chunk_id_必须是_str(value):
    with pytest.raises(TypeError, match="chunk_id 必须是 str"):
        _store.check_record(SCHEMA, good(chunk_id=value))


def test_记录必须是_mapping():
    with pytest.raises(TypeError, match="必须是 mapping"):
        _store.check_record(SCHEMA, ("c-1", [1, 0, 0, 0, 0, 0, 0, 0]))


def test_向量本身不在这里校验():
    """`check_record` 只看形状，向量交给 `_codec`。"""
    assert _store.check_record(SCHEMA, good(vector=None)) == "c-1"


# ---------------------------------------------------------------- SQL 构造

def test_insert_标识符加方括号():
    sql = _store.build_insert(SCHEMA)
    assert sql == (
        "insert into chunks "
        "([chunk_id], [embedding], [created_at], [updated_at], [loc], [content], [from]) "
        "values (?, ?, ?, ?, ?, ?, ?)"
    )


def test_search_带_distance_且用_k_约束():
    sql, values = _store.build_search(SCHEMA, [])
    assert sql == (
        "select [chunk_id], [created_at], [updated_at], [loc], [content], [from], "
        "[distance] from chunks where [embedding] match ? and k = ?"
    )
    assert values == []


def test_search_多键是_AND_并给出参数顺序():
    conditions = _store.check_where(SCHEMA, {"loc": "a.md", "chunk_id": "c-1"})
    sql, values = _store.build_search(SCHEMA, conditions)
    assert sql.endswith("where [embedding] match ? and [loc] = ? and [chunk_id] = ? and k = ?")
    assert values == ["a.md", "c-1"]  # 参数顺序由它决定


def test_search_范围子句按操作符拼接():
    conditions = _store.check_where(
        SCHEMA, {"loc": (">=", "2025-06-01"), "chunk_id": ("<", "c-9")}
    )
    sql, values = _store.build_search(SCHEMA, conditions)
    assert sql.endswith(
        "where [embedding] match ? and [loc] >= ? and [chunk_id] < ? and k = ?"
    )
    assert values == ["2025-06-01", "c-9"]


def test_get_不带_distance():
    sql = _store.build_get(SCHEMA)
    assert sql == (
        "select [chunk_id], [created_at], [updated_at], [loc], [content], [from] "
        "from chunks where [chunk_id] = ?"
    )


def test_update_的_SET_顺序就是参数顺序():
    sql = _store.build_update(["updated_at", "embedding", "loc"])
    assert sql == (
        "update chunks set [updated_at] = ?, [embedding] = ?, [loc] = ? "
        "where [chunk_id] = ?"
    )


def test_delete():
    assert _store.build_delete() == "delete from chunks where [chunk_id] = ?"


def test_所有_DML_都不用双引号():
    """双引号在找不到列名时会退化成字符串字面量（探针 19）。"""
    sqls = [
        _store.build_insert(SCHEMA),
        _store.build_search(SCHEMA, [("loc", "=", "x")])[0],
        _store.build_get(SCHEMA),
        _store.build_update(["updated_at"]),
        _store.build_delete(),
    ]
    for sql in sqls:
        assert '"' not in sql
        assert "`" not in sql
