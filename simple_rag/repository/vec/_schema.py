"""Schema 校验与 DDL 生成（纯函数）。

DDL 是字符串拼接，而 **vec0 在 DDL 里拒绝任何带引号的标识符**（实测四种引号风格
全被拒，探针 19）—— 引号这条路堵死了，字段名只能模块自己校验。

这里只管 **Schema 本身**合不合法；「一条记录相对 Schema 合不合法」是 `_store` 的事。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

# 一个 .db 文件一张表，表名由模块内部固定
TABLE_NAME = "chunks"

# 实测：前导下划线被拒（`Could not parse '_loc text'`）；含空格逗号会让 DDL
# **静默拆成多列**（`body text, extra` 真建出 body 与 extra 两列）
FIELD_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

# 库级拒绝 —— 实测这些名字在**建表时**就报 `duplicate column name`
RESERVED_BY_LIB = frozenset({"chunk_id", "embedding", "distance", "k"})
# 模块策略 —— 库其实允许这些列名，是模块自己要用它们
RESERVED_BY_MODULE = frozenset({"created_at", "updated_at", "vector"})
RESERVED = RESERVED_BY_LIB | RESERVED_BY_MODULE

METRICS = ("L2", "cosine")

# 实测边界（探针 18）：0 -> `could not parse vector column`；8193 -> too large
DIM_MIN, DIM_MAX = 1, 8192

# 实测（探针 18）：两类列**各自 16 上限、互不叠加**（16+16 可以同时满）。
# 两个时间戳占掉 2 个普通列，所以可过滤字段还剩 16 - 2 = 14 个名额。
MAX_FILTERABLE = 14
MAX_AUXILIARY = 16

# sqlite_master.sql 里存着完整 DDL，从里面读回 dim 与 metric（实测探针 18 §8.8）
_DIM_RE = re.compile(r"embedding\s+float\[\s*(\d+)\s*\]", re.IGNORECASE)
_METRIC_RE = re.compile(r"distance_metric\s*=\s*(\w+)", re.IGNORECASE)


@dataclass(frozen=True)
class Schema:
    """一个向量表的形状。

    **模块不认识这些字段代表的任何含义** —— 叫什么、存什么、要不要过滤，
    都是调用方的决定。
    """

    # 向量维度；建表时写进 DDL 的 float[dim]，之后改不了
    dim: int
    # 距离度量；同样在建表时锁死（见 _validate 里不给默认值的理由）
    metric: Literal["L2", "cosine"]
    # 字段名，顺序即建表时列的顺序；分成可过滤与不可过滤两类（见 build_ddl）
    fields: tuple[str, ...] = ()
    # filterable 必须是 fields 的子集：这些列声明为可过滤，能进 KNN search （向量检索时的条件筛选列）的 where
    # sqlite-vec中的普通列（metadata）
    filterable: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # 校验放在这里，非法的 Schema 根本构造不出来
        _validate(self)


def _validate(schema: Schema) -> None:
    if isinstance(schema.dim, bool) or not isinstance(schema.dim, int):
        raise ValueError(f"dim 必须是 int，收到 {type(schema.dim).__name__}")
    if not DIM_MIN <= schema.dim <= DIM_MAX:
        raise ValueError(f"dim 必须在 {DIM_MIN}~{DIM_MAX} 之间，收到 {schema.dim}")

    # metric 不给默认值：它在建表时锁死、之后改不了，给默认值等于替调用方拍板
    if schema.metric not in METRICS:
        raise ValueError(f"metric 只能是 {METRICS} 之一，收到 {schema.metric!r}")

    seen: dict[str, str] = {}
    for name in schema.fields:
        _check_field_name(name)
        # 实测：loc + LOC -> `duplicate column name: LOC`（SQLite 标识符大小写不敏感）
        key = name.lower()
        if key in seen:
            raise ValueError(f"字段名重复（大小写不敏感）：{seen[key]!r} 与 {name!r}")
        seen[key] = name

    filterable = set(schema.filterable)
    unknown = filterable - set(schema.fields)
    if unknown:
        raise ValueError(f"filterable 必须是 fields 的子集，多出：{sorted(unknown)}")
    for name in schema.filterable:
        _check_field_name(name)

    if len(filterable) > MAX_FILTERABLE:
        raise ValueError(
            f"可过滤字段最多 {MAX_FILTERABLE} 个"
            f"（普通列上限 16，被 created_at / updated_at 占掉 2 个），"
            f"收到 {len(filterable)}"
        )
    auxiliary = len(schema.fields) - len(filterable)
    if auxiliary > MAX_AUXILIARY:
        raise ValueError(f"其余字段（加号列）最多 {MAX_AUXILIARY} 个，收到 {auxiliary}")


def _check_field_name(name) -> None:
    if not isinstance(name, str):
        raise ValueError(f"字段名必须是 str，收到 {type(name).__name__}")
    if name in RESERVED:
        raise ValueError(f"字段名 {name!r} 是保留名，不能用：{sorted(RESERVED)}")
    if not FIELD_NAME_RE.match(name):
        raise ValueError(
            f"字段名 {name!r} 不合法：必须字母开头，后跟字母 / 数字 / 下划线"
        )


def build_ddl(schema: Schema) -> str:
    """生成建表语句。

    字段名**裸写** —— vec0 在 DDL 里拒绝一切带引号的标识符（探针 19）。
    """
    columns = [
        "chunk_id text primary key",
        f"embedding float[{schema.dim}] distance_metric={schema.metric}",
        "created_at text",
        "updated_at text",
    ]
    filterable = set(schema.filterable)
    # 显式声明可过滤的 -> 普通列，能进 search 的 where
    columns += [f"{name} text" for name in schema.fields if name in filterable]
    # 其余（默认）-> 加号列，不能在 search 的 where 里用，但适合放长文本
    columns += [f"+{name} text" for name in schema.fields if name not in filterable]
    return (
        f"create virtual table {TABLE_NAME} using vec0(\n    "
        + ",\n    ".join(columns)
        + "\n)"
    )


def parse_ddl(sql: str) -> tuple[int, str]:
    """从已有的 `sqlite_master.sql` 里读回 `(dim, metric)`。

    **不带扩展的连接也能读**（实测探针 18 §8.8）。
    """
    dim = _DIM_RE.search(sql)
    metric = _METRIC_RE.search(sql)
    if dim is None or metric is None:
        raise ValueError(f"这个库不是本模块建的，读不出 dim / metric：{sql!r}")
    return int(dim.group(1)), metric.group(1)
