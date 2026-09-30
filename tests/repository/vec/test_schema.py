"""`_schema` 的纯单测 —— 不碰数据库。"""

import pytest

from simple_rag.repository.vec._schema import (
    MAX_AUXILIARY,
    MAX_FILTERABLE,
    TABLE_NAME,
    Schema,
    build_ddl,
    parse_ddl,
)


def make(**kwargs) -> Schema:
    base = {"dim": 8, "metric": "cosine"}
    return Schema(**{**base, **kwargs})


# ---------------------------------------------------------------- Schema 基本

def test_最简构造():
    schema = make()
    assert schema.fields == ()
    assert schema.filterable == ()


def test_metric_没有默认值():
    with pytest.raises(TypeError):
        Schema(dim=8)


@pytest.mark.parametrize("dim", [1, 768, 8192])
def test_dim_合法边界(dim):
    assert make(dim=dim).dim == dim


@pytest.mark.parametrize("dim", [0, -1, 8193, 100000])
def test_dim_越界(dim):
    with pytest.raises(ValueError, match="dim 必须在"):
        make(dim=dim)


def test_dim_必须是_int():
    with pytest.raises(ValueError, match="dim 必须是 int"):
        make(dim="768")
    with pytest.raises(ValueError, match="dim 必须是 int"):
        make(dim=True)


@pytest.mark.parametrize("metric", ["l2", "L1", "euclidean", "", None])
def test_metric_只认_L2_和_cosine(metric):
    with pytest.raises(ValueError, match="metric 只能是"):
        make(metric=metric)


# ---------------------------------------------------------------- 字段名规则

def test_正常字段名():
    schema = make(fields=("loc", "content_2", "A"))
    assert schema.fields == ("loc", "content_2", "A")


@pytest.mark.parametrize(
    "name",
    [
        "_loc",           # 前导下划线
        "2loc",           # 数字开头
        "lo c",           # 空格
        "body text, extra",  # 逗号（会让 DDL 静默拆成多列）
        '"loc"',          # 引号
        "lo-c",           # 连字符
        "",               # 空
    ],
)
def test_非法字段名(name):
    with pytest.raises(ValueError):
        make(fields=(name,))


@pytest.mark.parametrize("name", [123, None, b"loc"])
def test_字段名必须是_str(name):
    with pytest.raises(ValueError, match="必须是 str"):
        make(fields=(name,))


@pytest.mark.parametrize("name", ["chunk_id", "embedding", "distance", "k"])
def test_库级保留名(name):
    with pytest.raises(ValueError, match="保留名"):
        make(fields=(name,))


@pytest.mark.parametrize("name", ["created_at", "updated_at", "vector"])
def test_模块保留名(name):
    with pytest.raises(ValueError, match="保留名"):
        make(fields=(name,))


def test_字段名大小写不敏感地重复():
    with pytest.raises(ValueError, match="重复"):
        make(fields=("loc", "LOC"))


# ---------------------------------------------------------------- filterable

def test_filterable_必须是_fields_子集():
    with pytest.raises(ValueError, match="子集"):
        make(fields=("loc",), filterable=("content",))


def test_filterable_元素也要过名字校验():
    with pytest.raises(ValueError):
        make(fields=("loc", "_bad"), filterable=("_bad",))


def test_filterable_正好_14_个可以():
    fields = tuple(f"f{i}" for i in range(MAX_FILTERABLE))
    assert len(make(fields=fields, filterable=fields).filterable) == MAX_FILTERABLE


def test_filterable_超过_14_个报错():
    fields = tuple(f"f{i}" for i in range(MAX_FILTERABLE + 1))
    with pytest.raises(ValueError, match="可过滤字段最多"):
        make(fields=fields, filterable=fields)


def test_其余字段正好_16_个可以():
    fields = tuple(f"f{i}" for i in range(MAX_AUXILIARY))
    assert len(make(fields=fields).fields) == MAX_AUXILIARY


def test_其余字段超过_16_个报错():
    fields = tuple(f"f{i}" for i in range(MAX_AUXILIARY + 1))
    with pytest.raises(ValueError, match="加号列"):
        make(fields=fields)


def test_两类列可以同时满():
    """实测（探针 18）：普通列 16 与加号列 16 互不叠加，可以同时满。

    普通列 = chunk_id + created_at + updated_at + 14 个可过滤字段 = 16。
    """
    filterable = tuple(f"a{i}" for i in range(MAX_FILTERABLE))
    auxiliary = tuple(f"b{i}" for i in range(MAX_AUXILIARY))
    schema = make(fields=filterable + auxiliary, filterable=filterable)
    ddl = build_ddl(schema)
    assert ddl.count(" text") == 3 + MAX_FILTERABLE + MAX_AUXILIARY
    assert ddl.count("+b") == MAX_AUXILIARY


# ---------------------------------------------------------------- DDL 生成

def test_ddl_字段名裸写():
    ddl = build_ddl(make(dim=768, metric="L2", fields=("loc", "content"), filterable=("loc",)))
    assert ddl == (
        f"create virtual table {TABLE_NAME} using vec0(\n"
        "    chunk_id text primary key,\n"
        "    embedding float[768] distance_metric=L2,\n"
        "    created_at text,\n"
        "    updated_at text,\n"
        "    loc text,\n"
        "    +content text\n"
        ")"
    )


def test_ddl_没有可过滤字段时不产生普通列段():
    ddl = build_ddl(make(fields=("content",)))
    assert "created_at text,\n    updated_at text,\n    +content text" in ddl
    assert "content text" in ddl and "+content text" in ddl
    assert "\n    content text," not in ddl


def test_ddl_没有字段时只有四列():
    ddl = build_ddl(make(dim=4, metric="cosine"))
    assert ddl.count(" text") == 3  # chunk_id / created_at / updated_at
    assert "+" not in ddl


def test_ddl_里没有任何引号():
    """vec0 在 DDL 里拒绝一切带引号的标识符（探针 19）。

    注意 `float[8]` 的方括号是 vec0 自己的维度语法，不算标识符引号。
    """
    ddl = build_ddl(make(fields=("loc", "content"), filterable=("loc",)))
    assert "'" not in ddl and '"' not in ddl and "`" not in ddl


# ---------------------------------------------------------------- parse_ddl

def test_parse_ddl_往返():
    for dim, metric in ((4, "cosine"), (768, "L2"), (8192, "cosine")):
        schema = make(dim=dim, metric=metric, fields=("loc",), filterable=("loc",))
        assert parse_ddl(build_ddl(schema)) == (dim, metric)


def test_parse_ddl_读得回大写关键字():
    """sqlite_master 里存的是 `CREATE VIRTUAL TABLE ...`，大小写被改写过。"""
    sql = (
        "CREATE VIRTUAL TABLE chunks using vec0(chunk_id text primary key, "
        "embedding float[32] distance_metric=cosine, created_at text)"
    )
    assert parse_ddl(sql) == (32, "cosine")


def test_parse_ddl_不认识的结构报错():
    with pytest.raises(ValueError, match="不是本模块建的"):
        parse_ddl("CREATE TABLE chunks (id integer, name text)")
