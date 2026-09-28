"""DML 的 SQL 文本构造 + 单条记录 / where / k 相对 Schema 的校验（纯函数）。

**所有标识符一律加方括号。** 实测（探针 19 测业务字段名、探针 21 补测库内建列）：
DML 里方括号对所有标识符都可用，且拼错列名会报 `no such column`。

⚠️ **双引号绝对不能用** —— SQLite 的双引号在找不到列名时会**退化成字符串字面量**，
拼错列名不报错，而是静默拿到一个常量。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ._schema import TABLE_NAME, Schema

# 实测边界（探针 18）：4097 -> `k value in knn query too large ... the limit is 4096`
K_MIN, K_MAX = 0, 4096

# KNN 的 where 支持的比较操作符。实测（探针 06 / 18）：metadata 列在 KNN 里支持
# 等值 / 范围 / IN / OR —— 本模块只开放等值与范围比较（用途是时间范围）。
# `=` 与直接给 str 等价。
_COMPARISON_OPS = frozenset({"=", ">", ">=", "<", "<="})

# `check_where` 归一化后的一条条件：(字段, 操作符, 值)
Condition = tuple[str, str, str]


def quote(name: str) -> str:
    return f"[{name}]"


def _identifiers(names: Sequence[str]) -> str:
    return ", ".join(quote(name) for name in names)


def _select_names(schema: Schema, *, with_distance: bool) -> list[str]:
    names = ["chunk_id", "created_at", "updated_at", *schema.fields]
    if with_distance:
        names.append("distance")
    return names


def build_insert(schema: Schema) -> str:
    names = ["chunk_id", "embedding", "created_at", "updated_at", *schema.fields]
    placeholders = ", ".join("?" for _ in names)
    return f"insert into {TABLE_NAME} ({_identifiers(names)}) values ({placeholders})"


def build_search(
    schema: Schema, conditions: Sequence[Condition]
) -> tuple[str, list[str]]:
    """返回 `(sql, where 的参数值列表)` —— 参数顺序由第二个值决定。

    `conditions` 是 `check_where` 归一化后的 `(字段, 操作符, 值)` 序列；
    操作符来自固定白名单，直接拼进 SQL 是安全的。
    """
    clauses = [f"{quote('embedding')} match ?"]
    values: list[str] = []
    for key, op, value in conditions:
        clauses.append(f"{quote(key)} {op} ?")
        values.append(value)
    sql = (
        f"select {_identifiers(_select_names(schema, with_distance=True))} "
        f"from {TABLE_NAME} where {' and '.join(clauses)} and k = ?"
    )
    return sql, values


def build_get(schema: Schema) -> str:
    """普通查询：**不带 `distance`**。

    实测（探针 21）非 KNN 查询里 `select distance` 不报错、给 NULL，
    但那是未文档化的巧合，不依赖它。
    """
    return (
        f"select {_identifiers(_select_names(schema, with_distance=False))} "
        f"from {TABLE_NAME} where {quote('chunk_id')} = ?"
    )


def build_update(columns: Sequence[str]) -> str:
    """`columns` 是 SET 子句里的列名，顺序即参数顺序。

    调用方**绝不能**把 `chunk_id` 放进来 —— 实测库会报
    `UPDATEs on vec0 primary key values are not allowed.`
    """
    assignments = ", ".join(f"{quote(name)} = ?" for name in columns)
    return f"update {TABLE_NAME} set {assignments} where {quote('chunk_id')} = ?"


def build_delete() -> str:
    return f"delete from {TABLE_NAME} where {quote('chunk_id')} = ?"


def check_k(k) -> int:
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k 必须是 int，收到 {type(k).__name__}")
    if not K_MIN <= k <= K_MAX:
        raise ValueError(f"k 必须在 {K_MIN}~{K_MAX} 之间，收到 {k}")
    return k


def check_where(schema: Schema, where) -> list[Condition]:
    """校验并归一化 `search` 的 `where`，返回 `(字段, 操作符, 值)` 列表（多项 AND）。

    key 必须是 `chunk_id` 或显式声明为可过滤的字段 —— 其余字段存成加号列，
    在 KNN 里过滤不了（实测探针 18）。

    value 两种写法：

    - `str`          —— 等值，等价于 `("=", value)`
    - `(op, value)`  —— 比较，`op` 是 `=` / `>` / `>=` / `<` / `<=` 之一

    ⚠️ 字段列都是 `TEXT`，比较是**字符串序**：对 ISO 时间戳（等长，字典序即时间序）
    正确，对纯数字不正确（`"9" >= "18"` 为真）。
    """
    if where is None:
        return []
    if not isinstance(where, Mapping):
        raise TypeError(f"where 必须是 mapping，收到 {type(where).__name__}")
    allowed = {"chunk_id", *schema.filterable}
    conditions: list[Condition] = []
    for key, value in where.items():
        if key not in allowed:
            raise ValueError(
                f"where 的键 {key!r} 不能用于过滤：只能是 'chunk_id'，"
                f"或 Schema.filterable 里声明的字段 {list(schema.filterable)}"
            )
        op, raw = _parse_condition(key, value)
        conditions.append((key, op, raw))
    return conditions


def _parse_condition(key: str, value) -> tuple[str, str]:
    """把一个 `where` 的值归一化成 `(操作符, 文本值)`。"""
    if isinstance(value, str):
        return "=", value
    if isinstance(value, tuple) and len(value) == 2:
        op, raw = value
        if op not in _COMPARISON_OPS:
            raise ValueError(
                f"where 的键 {key!r} 的比较操作符 {op!r} 不支持："
                f"只能是 {sorted(_COMPARISON_OPS)} 之一，或直接给 str 表示等值"
            )
        if not isinstance(raw, str):
            raise TypeError(
                f"where 的键 {key!r} 的比较值必须是 str，收到 {type(raw).__name__}"
            )
        return op, raw
    raise TypeError(
        f"where 的值必须是 str（等值）或 (操作符, 值) 元组，"
        f"键 {key!r} 收到 {type(value).__name__}"
    )


def check_record(schema: Schema, record) -> str:
    """校验一条记录的形状，返回它的 `chunk_id`。

    向量本身由 `_codec` 管，这里不碰。
    """
    if not isinstance(record, Mapping):
        raise TypeError(f"每条记录必须是 mapping，收到 {type(record).__name__}")

    keys = set(record)
    expected = {"chunk_id", "vector", *schema.fields}
    unknown = keys - expected
    if unknown:
        # 包括 created_at / updated_at（模块管理的字段）和拼错的字段名 —— 不静默忽略
        raise ValueError(
            f"记录里有不认识的键 {sorted(unknown)}，本 Schema 只接受 {sorted(expected)}"
        )
    for name in ("chunk_id", "vector"):
        if name not in keys:
            raise ValueError(f"记录缺必需的键 {name!r}")

    chunk_id = record["chunk_id"]
    if not isinstance(chunk_id, str):
        raise TypeError(f"chunk_id 必须是 str，收到 {type(chunk_id).__name__}")

    for name in schema.fields:
        # 加号列不写会读回 None（实测探针 18），但模块要求输入形状统一 ——
        # 这是模块的策略，不是库的要求
        if name not in keys:
            raise ValueError(f"记录缺已声明的字段 {name!r}")
        value = record[name]
        if value is not None and not isinstance(value, str):
            raise TypeError(
                f"字段 {name!r} 的值必须是 str 或 None，收到 {type(value).__name__}"
            )
    return chunk_id
