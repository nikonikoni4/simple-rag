"""`VecStore` 的端到端测试 —— 真开临时库，真跑 vec0。"""

import sqlite3

import pytest

from simple_rag.repository import Schema, VecSearchResult, VecStore
from simple_rag.repository._schema import MAX_AUXILIARY, MAX_FILTERABLE, TABLE_NAME

DIM = 8

SCHEMA = Schema(dim=DIM, metric="cosine", fields=("loc", "content"), filterable=("loc",))

# 字段名故意全用 SQL 关键字（实测探针 19：DDL 里能建、DML 里必须加方括号）
KEYWORD_SCHEMA = Schema(
    dim=DIM,
    metric="cosine",
    fields=("from", "order", "select", "where"),
    filterable=("from", "order"),
)


def onehot(i: int) -> list[float]:
    vector = [0.0] * DIM
    vector[i] = 1.0
    return vector


def ramp(i: int) -> list[float]:
    """前 i+1 维为 1。[1,0,...] 与 ramp(i) 的余弦距离随 i 严格递增。"""
    return [1.0 if j <= i else 0.0 for j in range(DIM)]


def record(chunk_id: str, *, vector=None, loc="a.md", content="正文") -> dict:
    return {
        "chunk_id": chunk_id,
        "vector": onehot(0) if vector is None else vector,
        "loc": loc,
        "content": content,
    }


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "vec.db"


@pytest.fixture
def store(db_path):
    instance = VecStore(db_path)
    yield instance
    instance.close()


# ---------------------------------------------------------------- 基本流程

def test_增删改查走一遍(store):
    store.create_table(SCHEMA)
    store.insert([record("c-1"), record("c-2", loc="b.md")])

    hits = store.search(onehot(0), k=5)
    assert {h.chunk_id for h in hits} == {"c-1", "c-2"}
    assert all(h.distance is not None for h in hits)

    row = store.get("c-2")
    assert isinstance(row, VecSearchResult)  # get 与 search 返回同一个类型
    assert row.chunk_id == "c-2"
    assert row.distance is None  # get 不是相似度查询
    assert row.fields == {"loc": "b.md", "content": "正文"}

    assert store.update("c-1", vector=onehot(3), fields={"content": "改过"}) == 1
    assert store.get("c-1").fields["content"] == "改过"

    assert store.delete("c-2") == 1
    assert store.get("c-2") is None


def test_with_语句(store):
    with store as s:
        s.create_table(SCHEMA)
        s.insert([record("c-1")])
    # 退出 with 之后连接已关
    with pytest.raises(sqlite3.ProgrammingError):
        store.get("c-1")


def test_重复_close_是_noop(store):
    store.create_table(SCHEMA)
    store.close()
    store.close()
    store.close()


def test_没调_create_table_就操作报错(store):
    with pytest.raises(ValueError, match="还没调 create_table"):
        store.insert([record("c-1")])
    with pytest.raises(ValueError, match="还没调 create_table"):
        store.search(onehot(0), k=1)
    with pytest.raises(ValueError, match="还没调 create_table"):
        store.get("c-1")
    with pytest.raises(ValueError, match="还没调 create_table"):
        store.update("c-1", vector=onehot(0))
    with pytest.raises(ValueError, match="还没调 create_table"):
        store.delete("c-1")


def test_关闭后再操作抛_ProgrammingError(store):
    store.create_table(SCHEMA)
    store.close()
    with pytest.raises(sqlite3.ProgrammingError):
        store.get("c-1")


# ---------------------------------------------------------------- 重开与建表校验

def test_重开同一个库不报_table_already_exists(db_path):
    first = VecStore(db_path)
    first.create_table(SCHEMA)
    first.insert([record("c-1")])
    first.close()

    second = VecStore(db_path)
    second.create_table(SCHEMA)  # 最常规的用法：每次开库都调
    assert second.get("c-1").fields["loc"] == "a.md"
    second.close()


@pytest.mark.parametrize(
    "other",
    [
        Schema(dim=DIM * 2, metric="cosine"),
        Schema(dim=DIM, metric="L2"),
    ],
)
def test_打开的不是这个_Schema_建的库(db_path, other):
    first = VecStore(db_path)
    first.create_table(SCHEMA)
    first.close()

    second = VecStore(db_path)
    with pytest.raises(ValueError, match="不是这个 Schema 建的库"):
        second.create_table(other)
    second.close()


def test_库里已有的表不是本模块建的(db_path):
    plain = sqlite3.connect(db_path)
    plain.execute(f"create table {TABLE_NAME} (id integer, name text)")
    plain.commit()
    plain.close()

    store = VecStore(db_path)
    with pytest.raises(ValueError, match="不是本模块建的"):
        store.create_table(SCHEMA)
    store.close()


def test_没建表时库里没有表(store):
    store.create_table(SCHEMA)
    sql = store._read_ddl()
    assert sql is not None
    assert f"float[{DIM}]" in sql
    assert "distance_metric=cosine" in sql
    assert "+content text" in sql  # 没声明可过滤的字段存成加号列


# ---------------------------------------------------------------- chunk_id 唯一性

def test_与库里已有的_chunk_id_冲突_整批回滚(store):
    store.create_table(SCHEMA)
    store.insert([record("c-1")])

    with pytest.raises(ValueError, match="chunk_id 重复"):
        store.insert([record("c-2"), record("c-1"), record("c-3")])

    # 实测（探针 18 §8.5 / 21 §11.5）：不显式 rollback 的话半批会真的落库
    assert store.get("c-2") is None
    assert store.get("c-3") is None
    assert store.get("c-1") is not None


def test_批内重复的_chunk_id_也整批回滚(store):
    store.create_table(SCHEMA)
    with pytest.raises(ValueError, match="chunk_id 重复"):
        store.insert([record("c-1"), record("c-2"), record("c-1")])
    assert store.get("c-1") is None
    assert store.get("c-2") is None


def test_回滚之后还能正常写(store):
    """冲突不该把连接卡在某个坏状态里。"""
    store.create_table(SCHEMA)
    store.insert([record("c-1")])
    with pytest.raises(ValueError):
        store.insert([record("c-1")])
    store.insert([record("c-2")])
    assert store.get("c-2") is not None


# ---------------------------------------------------------------- insert 的校验

def test_空输入直接返回(store):
    store.create_table(SCHEMA)
    assert store.insert([]) is None
    assert store.insert(iter([])) is None


def test_生成器入参只被消费一次(store):
    """入参是可迭代对象，生成器被遍历两遍的话第二遍是空的、会静默插 0 条。"""
    store.create_table(SCHEMA)
    store.insert(record(f"c-{i}") for i in range(3))
    assert len(store.search(onehot(0), k=10)) == 3


def test_校验失败时一条都不写(store):
    store.create_table(SCHEMA)
    bad = record("c-2")
    del bad["content"]
    with pytest.raises(ValueError, match="缺已声明的字段"):
        store.insert([record("c-1"), bad])
    assert store.get("c-1") is None  # 校验在开事务之前完成


@pytest.mark.parametrize(
    "mutate, exc",
    [
        (lambda r: r.pop("chunk_id"), ValueError),
        (lambda r: r.pop("vector"), ValueError),
        (lambda r: r.pop("loc"), ValueError),
        (lambda r: r.update(created_at="x"), ValueError),
        (lambda r: r.update(updated_at="x"), ValueError),
        (lambda r: r.update(nosuchfield="x"), ValueError),
        (lambda r: r.update(chunk_id=123), TypeError),
        (lambda r: r.update(loc=123), TypeError),
        (lambda r: r.update(vector={"a": 1}), TypeError),
        (lambda r: r.update(vector=[1.0, 2.0]), ValueError),  # 维度不符
    ],
)
def test_insert_的逐条校验(store, mutate, exc):
    store.create_table(SCHEMA)
    bad = record("c-1")
    mutate(bad)
    with pytest.raises(exc):
        store.insert([bad])


def test_向量含_NaN_被拦且不写库(store):
    store.create_table(SCHEMA)
    with pytest.raises(ValueError, match="NaN"):
        store.insert([record("c-1", vector=[float("nan")] * DIM)])
    assert store.get("c-1") is None


# ---------------------------------------------------------------- 时间戳

def test_insert_自动填时间戳_两者同值(store):
    store.create_table(SCHEMA)
    store.insert([record("c-1")])
    row = store.get("c-1")
    assert row.created_at == row.updated_at
    assert row.created_at.endswith("+00:00")
    assert "T" in row.created_at


def test_update_刷新_updated_at_但不动_created_at(store):
    store.create_table(SCHEMA)
    store.insert([record("c-1")])
    before = store.get("c-1")

    store.update("c-1", fields={"content": "改过"})
    after = store.get("c-1")

    assert after.created_at == before.created_at
    assert after.updated_at >= before.updated_at


def test_不返回向量(store):
    """`get` 拿不到向量 —— 已知代价，见 spec §12。"""
    store.create_table(SCHEMA)
    store.insert([record("c-1")])
    assert not hasattr(store.get("c-1"), "vector")


# ---------------------------------------------------------------- search

def test_search_按距离严格升序(store):
    store.create_table(SCHEMA)
    store.insert([record(f"c-{i}", vector=ramp(i)) for i in range(4)])

    hits = store.search(onehot(0), k=4)
    assert [h.chunk_id for h in hits] == ["c-0", "c-1", "c-2", "c-3"]
    distances = [h.distance for h in hits]
    assert distances == sorted(distances)
    assert hits[0].distance == pytest.approx(0.0, abs=1e-6)


def test_search_的_k_限制条数(store):
    store.create_table(SCHEMA)
    store.insert([record(f"c-{i}", vector=ramp(i)) for i in range(5)])
    assert len(store.search(onehot(0), k=2)) == 2
    assert store.search(onehot(0), k=0) == []


def test_search_的_where_过滤生效(store):
    store.create_table(SCHEMA)
    store.insert([
        record("c-1", loc="a.md"),
        record("c-2", loc="b.md"),
        record("c-3", loc="a.md"),
    ])
    hits = store.search(onehot(0), k=10, where={"loc": "a.md"})
    assert {h.chunk_id for h in hits} == {"c-1", "c-3"}


def test_search_的_where_可以带_chunk_id(store):
    store.create_table(SCHEMA)
    store.insert([record("c-1"), record("c-2")])
    hits = store.search(onehot(0), k=10, where={"chunk_id": "c-2"})
    assert [h.chunk_id for h in hits] == ["c-2"]


def test_search_的_where_多键是_AND(store):
    store.create_table(SCHEMA)
    store.insert([record("c-1", loc="a.md"), record("c-2", loc="a.md")])
    hits = store.search(onehot(0), k=10, where={"loc": "a.md", "chunk_id": "c-1"})
    assert [h.chunk_id for h in hits] == ["c-1"]


def test_search_不能过滤加号列(store):
    store.create_table(SCHEMA)  # content 是加号列
    store.insert([record("c-1")])
    with pytest.raises(ValueError, match="不能用于过滤"):
        store.search(onehot(0), k=10, where={"content": "正文"})


def test_search_的_where_None_和空_dict_等价(store):
    store.create_table(SCHEMA)
    store.insert([record("c-1"), record("c-2")])
    assert len(store.search(onehot(0), k=10, where=None)) == 2
    assert len(store.search(onehot(0), k=10, where={})) == 2


@pytest.mark.parametrize("k", [-1, 4097])
def test_search_的_k_越界(store, k):
    store.create_table(SCHEMA)
    with pytest.raises(ValueError, match="k 必须在"):
        store.search(onehot(0), k=k)


def test_search_的_k_不接受_bool(store):
    store.create_table(SCHEMA)
    with pytest.raises(TypeError, match="k 必须是 int"):
        store.search(onehot(0), k=True)


def test_search_的查询向量也归一化(store):
    """两端共用同一套编码，长度不同的同向向量结果一致。"""
    store.create_table(SCHEMA)
    store.insert([record(f"c-{i}", vector=[x * 100 for x in ramp(i)]) for i in range(3)])
    hits = store.search([0.001 * x for x in onehot(0)], k=3)
    assert [h.chunk_id for h in hits] == ["c-0", "c-1", "c-2"]


# ---------------------------------------------------------------- get

def test_get_不存在的_id_返回_None(store):
    store.create_table(SCHEMA)
    assert store.get("nope") is None


def test_get_能取到加号列(store):
    store.create_table(SCHEMA)
    store.insert([record("c-1", content="加号列里的长文本" * 100)])
    assert store.get("c-1").fields["content"] == "加号列里的长文本" * 100


def test_get_的_chunk_id_必须是_str(store):
    store.create_table(SCHEMA)
    with pytest.raises(TypeError, match="chunk_id 必须是 str"):
        store.get(123)


def test_加号列不写读回_None(store):
    """实测（探针 18）：普通列不能为 NULL，加号列可以。"""
    store.create_table(SCHEMA)
    store.insert([record("c-1", content=None)])
    assert store.get("c-1").fields["content"] is None


def test_普通列不能为_None_且失败不留半批(store):
    """`loc` 是可过滤字段（普通列），传 None 由库拦下。

    模块不校验这条 —— 报错明确（`Expected text for TEXT metadata column`），
    属于「缺了还能工作」；这里只确认它**不会留下半批数据**。
    """
    store.create_table(SCHEMA)
    with pytest.raises(sqlite3.Error):
        store.insert([record("c-1"), record("c-2", loc=None)])
    assert store.get("c-1") is None
    assert store.get("c-2") is None


# ---------------------------------------------------------------- update

def test_update_只改给出的字段(store):
    store.create_table(SCHEMA)
    store.insert([record("c-1", loc="a.md", content="原")])
    store.update("c-1", fields={"content": "新"})
    row = store.get("c-1")
    assert row.fields == {"loc": "a.md", "content": "新"}


def test_update_可以只改向量(store):
    store.create_table(SCHEMA)
    store.insert([record("c-1", vector=onehot(0))])
    assert store.update("c-1", vector=onehot(5)) == 1
    hits = store.search(onehot(5), k=1)
    assert hits[0].distance == pytest.approx(0.0, abs=1e-6)


def test_update_可以改加号列(store):
    store.create_table(SCHEMA)
    store.insert([record("c-1")])
    store.update("c-1", fields={"content": "改加号列"})
    assert store.get("c-1").fields["content"] == "改加号列"


def test_update_不存在的_id_返回_0(store):
    store.create_table(SCHEMA)
    assert store.update("nope", vector=onehot(0)) == 0


def test_update_什么都不给就报错(store):
    store.create_table(SCHEMA)
    store.insert([record("c-1")])
    with pytest.raises(ValueError, match="至少要给"):
        store.update("c-1")
    with pytest.raises(ValueError, match="至少要给"):
        store.update("c-1", fields={})


def test_update_不许改_chunk_id(store):
    store.create_table(SCHEMA)
    store.insert([record("c-1")])
    with pytest.raises(ValueError, match="不是已声明的字段"):
        store.update("c-1", fields={"chunk_id": "c-2"})


@pytest.mark.parametrize("name", ["created_at", "updated_at"])
def test_update_不许改模块管理的字段(store, name):
    store.create_table(SCHEMA)
    store.insert([record("c-1")])
    with pytest.raises(ValueError, match="不是已声明的字段"):
        store.update("c-1", fields={name: "2000-01-01T00:00:00+00:00"})


@pytest.mark.parametrize("name", ["nosuchfield", "Loc"])
def test_update_不许改未声明的字段(store, name):
    store.create_table(SCHEMA)
    store.insert([record("c-1")])
    with pytest.raises(ValueError, match="不是已声明的字段"):
        store.update("c-1", fields={name: "x"})


def test_update_字段值类型不对(store):
    store.create_table(SCHEMA)
    store.insert([record("c-1")])
    with pytest.raises(TypeError, match="必须是 str 或 None"):
        store.update("c-1", fields={"loc": 123})


def test_update_的_chunk_id_必须是_str(store):
    store.create_table(SCHEMA)
    with pytest.raises(TypeError, match="chunk_id 必须是 str"):
        store.update(123, vector=onehot(0))


def test_update_向量也走归一化(store):
    store.create_table(SCHEMA)
    store.insert([record("c-1", vector=onehot(0))])
    store.update("c-1", vector=[0.0, 100.0] + [0.0] * (DIM - 2))
    hits = store.search(onehot(1), k=1)
    assert hits[0].distance == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------- delete

def test_delete_不存在的_id_返回_0(store):
    store.create_table(SCHEMA)
    assert store.delete("nope") == 0


def test_delete_的_chunk_id_必须是_str(store):
    store.create_table(SCHEMA)
    with pytest.raises(TypeError, match="chunk_id 必须是 str"):
        store.delete(123)


def test_delete_之后可以再插回同样的_id(store):
    """实测（探针 20）：「改文档 = delete + insert 同 chunk_id」不泄漏。"""
    store.create_table(SCHEMA)
    for _ in range(3):
        store.insert([record("c-1")])
        assert store.delete("c-1") == 1
    store.insert([record("c-1")])
    assert len(store.search(onehot(0), k=10)) == 1


# ---------------------------------------------------------------- SQL 关键字字段名

def test_SQL_关键字字段名能正常增删改查(store):
    store.create_table(KEYWORD_SCHEMA)
    store.insert([
        {
            "chunk_id": "c-1",
            "vector": onehot(0),
            "from": "甲",
            "order": "乙",
            "select": "丙",
            "where": "丁",
        }
    ])

    hits = store.search(onehot(0), k=5, where={"from": "甲"})
    assert [h.chunk_id for h in hits] == ["c-1"]
    assert hits[0].fields == {"from": "甲", "order": "乙", "select": "丙", "where": "丁"}

    assert store.search(onehot(0), k=5, where={"order": "乙"})[0].chunk_id == "c-1"
    assert store.search(onehot(0), k=5, where={"from": "甲", "order": "乙"})[0].chunk_id == "c-1"
    assert store.search(onehot(0), k=5, where={"from": "别的"}) == []

    store.update("c-1", fields={"where": "戊"})
    assert store.get("c-1").fields["where"] == "戊"

    assert store.delete("c-1") == 1
    assert store.get("c-1") is None


def test_SQL_关键字字段名重开库照常(store, db_path):
    store.create_table(KEYWORD_SCHEMA)
    store.insert([{"chunk_id": "c-1", "vector": onehot(0), "from": "甲",
                   "order": "乙", "select": "丙", "where": "丁"}])
    store.close()

    again = VecStore(db_path)
    again.create_table(KEYWORD_SCHEMA)
    assert again.search(onehot(0), k=1, where={"from": "甲"})[0].fields["from"] == "甲"
    again.close()


# ---------------------------------------------------------------- 列数上限

def test_列数上限能真的建出来(store):
    """`MAX_FILTERABLE = 14` 必须**真建表**才算数。

    光数 DDL 字符串证明不了 —— 得让 vec0 自己承认。占满 16 个 metadata 列的是
    `created_at` + `updated_at` + 14 个可过滤字段；**`chunk_id` 是主键，不占名额**
    （实测：15 个可过滤字段就报 `More than 16 metadata columns were provided`）。
    """
    filterable = tuple(f"a{i}" for i in range(MAX_FILTERABLE))
    auxiliary = tuple(f"b{i}" for i in range(MAX_AUXILIARY))
    store.create_table(
        Schema(
            dim=DIM,
            metric="cosine",
            fields=filterable + auxiliary,
            filterable=filterable,
        )
    )

    record = {"chunk_id": "c-1", "vector": onehot(0)}
    record.update({name: f"v{i}" for i, name in enumerate(filterable + auxiliary)})
    store.insert([record])

    # 两类列都取得回来
    assert store.get("c-1").fields == {
        name: record[name] for name in filterable + auxiliary
    }
    # 可过滤的那批能进 where，加号列那批不能
    assert store.search(onehot(0), k=1, where={filterable[0]: "v0"})[0].chunk_id == "c-1"
    last = store.search(onehot(0), k=1, where={filterable[-1]: f"v{MAX_FILTERABLE - 1}"})
    assert [hit.chunk_id for hit in last] == ["c-1"]
    with pytest.raises(ValueError, match="不能用于过滤"):
        store.search(onehot(0), k=1, where={auxiliary[0]: f"v{MAX_FILTERABLE}"})


def test_超过列数上限在_Schema_阶段就被拦(store):
    """库那边报的是 `vec0 constructor error: More than 16 metadata columns`，
    模块不该把这种错留给库 —— 它是纯 schema 规则，本地就能判。"""
    too_many = tuple(f"a{i}" for i in range(MAX_FILTERABLE + 1))
    with pytest.raises(ValueError, match="可过滤字段最多"):
        Schema(dim=DIM, metric="cosine", fields=too_many, filterable=too_many)


# ---------------------------------------------------------------- 独立性

def test_可以同时开多个库(tmp_path):
    """连接由实例持有，不是模块级全局。"""
    a = VecStore(tmp_path / "a.db")
    b = VecStore(tmp_path / "b.db")
    try:
        a.create_table(SCHEMA)
        b.create_table(Schema(dim=4, metric="L2"))
        a.insert([record("c-1")])
        assert a.get("c-1") is not None
        assert b.get("c-1") is None
    finally:
        a.close()
        b.close()


def test_close_之后_db_文件可以删掉(db_path):
    """不关连接的话 Windows 上会 WinError 32（实测）。"""
    store = VecStore(db_path)
    store.create_table(SCHEMA)
    store.insert([record("c-1")])
    store.close()
    db_path.unlink()
    assert not db_path.exists()
