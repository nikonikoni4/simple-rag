"""表结构校验与 DDL 生成（纯函数）。

数据表是**普通 SQLite 表**，不是 vec0 那样的虚拟表。三处直接后果：

- DDL 里列名**一律用方括号括起来**（vec0 拒绝一切带引号的标识符，普通表没这个限制）
- 没有 16 列上限，也没有 metadata / 辅助列之分 —— 普通表的列一律能进 `where`
- 加列走 `ALTER TABLE`，不需要重建表，更不需要重跑 embedding

模块自己工作要用的列叫**基本字段**，DDL 里写死；调用方要加的列由 `Schema.fields`
声明。这条约定与 `repository/vec` 的 `Schema` 是同一套，区别只在于那边还要
额外区分 metadata / 辅助列。见 `simple_rag/repository/表结构约定.md`。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 一个 .db 文件一张表，表名由模块内部固定
TABLE_NAME = "chunk_data"

# 基本字段：写入时必须给，DDL 里固定
BASE_COLUMNS = (
    "chunk_id",
    "path",
    "start_line",
    "end_line",
    "created_at",
    "updated_at",
    "content",
)

# 不能再被声明成扩展列的名字：基本字段名 + 记录级的保留键
# （`sources` 不是表上的列，但它是记录里的键 —— 撞名会让报错信息指向错误的原因）
RESERVED = frozenset((*BASE_COLUMNS, "sources"))

FIELD_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class Schema:
    """数据表的形状：调用方要加的扩展列。

    **模块不认识这些列代表的任何含义** —— 叫什么、存什么，都是调用方的决定。
    与 vec0 的 `Schema` 不同，这里**没有 `filterable`**：普通表的每一列都能进
    `where`，不需要事先声明。
    """

    # 扩展列名，顺序即建表时的列顺序
    fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # 校验放在这里，非法的 Schema 根本构造不出来
        _validate(self)


def _validate(schema: Schema) -> None:
    seen: dict[str, str] = {}
    for name in schema.fields:
        _check_field_name(name)
        # SQLite 的标识符大小写不敏感，f1 + F1 会撞 `duplicate column name`
        key = name.lower()
        if key in seen:
            raise ValueError(f"字段名重复（大小写不敏感）：{seen[key]!r} 与 {name!r}")
        seen[key] = name


def _check_field_name(name) -> None:
    if not isinstance(name, str):
        raise ValueError(f"字段名必须是 str，收到 {type(name).__name__}")
    if name in RESERVED:
        raise ValueError(f"字段名 {name!r} 是保留名，不能当扩展列：{sorted(RESERVED)}")
    if not FIELD_NAME_RE.match(name):
        raise ValueError(
            f"字段名 {name!r} 不合法：必须字母开头，后跟字母 / 数字 / 下划线"
        )


def _all_columns(schema: Schema) -> tuple[str, ...]:
    """基本字段在前，扩展列在后 —— 建表、查询都用这个顺序。"""
    return (*BASE_COLUMNS, *schema.fields)


def _quote(name: str) -> str:
    """DDL 里把列名括起来。

    普通表接受方括号（`vec0` 不接受任何带引号的标识符 —— 那是 vec 那边的限制，
    普通表没有）。不括的话，`Schema(fields=("from",))` 这种把 SQL 关键字当列名
    的情况会直接建表失败。
    """
    return f"[{name}]"


def build_ddl(schema: Schema) -> str:
    """生成建表语句。

    基本字段的类型由模块定：`chunk_id` / `path` / `content` 与两个时间戳是
    `text`，行区间是 `integer`。扩展列一律 `text` —— 模块不替调用方决定类型，
    要别的类型就在建表后自己 `ALTER`。

    列名**一律方括号括起来**，理由见 `_quote`。
    """
    types = {
        "chunk_id": "text",
        "path": "text",
        "start_line": "integer",
        "end_line": "integer",
        "created_at": "text",
        "updated_at": "text",
        "content": "text",
    }
    columns = [
        f"{_quote(name)} {types.get(name, 'text')} not null" for name in BASE_COLUMNS
    ]
    # 扩展列允许为 NULL：调用方不总是每一列都有值
    columns += [f"{_quote(name)} text" for name in schema.fields]
    return f"create table {TABLE_NAME} (\n    " + ",\n    ".join(columns) + "\n)"


def build_indexes() -> list[str]:
    """两个索引，对应两条定位路径。

    - `(chunk_id)` —— 按 id 取回一个 chunk 的全部行（检索时回查正文与来源）
    - `(path)`    —— 按文件反查 chunk_id（更新文档时先定位要删哪些）
    """
    return [
        f"create index {TABLE_NAME}_chunk_id on {TABLE_NAME}(chunk_id)",
        f"create index {TABLE_NAME}_path on {TABLE_NAME}(path)",
    ]
