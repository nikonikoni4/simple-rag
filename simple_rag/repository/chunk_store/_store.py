"""DML 的 SQL 文本构造 + 记录校验（纯函数）。

**所有标识符一律加方括号。** 理由与 `repository/vec` 那边相同：
SQLite 的双引号在找不到列名时会**退化成字符串字面量**，拼错列名不报错，
而是静默拿到一个常量。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ._schema import BASE_COLUMNS, TABLE_NAME, Schema


def quote(name: str) -> str:
    return f"[{name}]"


def _columns(schema: Schema) -> list[str]:
    """基本字段在前、扩展列在后 —— 与建表顺序一致。"""
    return [*BASE_COLUMNS, *schema.fields]


def _identifiers(names: Sequence[str]) -> str:
    return ", ".join(quote(name) for name in names)


def build_insert(schema: Schema) -> str:
    names = _columns(schema)
    placeholders = ", ".join("?" for _ in names)
    return f"insert into {TABLE_NAME} ({_identifiers(names)}) values ({placeholders})"


def build_get(schema: Schema) -> str:
    """按 `chunk_id` 取回该 chunk 的**全部行**。

    `order by` 不是装饰：一个 chunk 有多行（多来源）时，顺序不稳定会让调用方
    拿到随机的来源顺序，进而让上层拼接结果、比对结果都跟着抖。
    """
    return (
        f"select {_identifiers(_columns(schema))} from {TABLE_NAME} "
        f"where {quote('chunk_id')} = ? "
        f"order by {quote('path')}, {quote('start_line')}"
    )


def build_ids_by_path() -> str:
    """按文件反查 `chunk_id` —— 删一个文件前，先问「哪些块来自它」。

    必须 `distinct`：同一个 chunk 在数据表里可能有多行（多来源），
    不去重的话调用方会拿到同一个 id 好几遍。
    """
    return (
        f"select distinct {quote('chunk_id')} from {TABLE_NAME} "
        f"where {quote('path')} = ?"
    )


def build_delete_by_chunk_id() -> str:
    return f"delete from {TABLE_NAME} where {quote('chunk_id')} = ?"


def build_delete_by_path() -> str:
    """按文件删——只动数据表自己的行，不碰向量表。

    删的是来自该文件的**行**。一个 chunk 若还有别的来源，它其余的行会留下；
    若这是它最后一行，这个 `chunk_id` 就在数据表里消失了。
    """
    return f"delete from {TABLE_NAME} where {quote('path')} = ?"


def check_record(schema: Schema, record) -> tuple[str, list[dict]]:
    """校验一条记录，返回 `(chunk_id, sources)`。

    一条记录 = **一个 chunk**，不是一行：`sources` 里的每项对应数据表的一行。
    这个形状与 `VecDB.insert` 对齐（那边一条记录也是一个 chunk），
    入库时两侧可以喂同一批对象。
    """
    if not isinstance(record, Mapping):
        raise TypeError(f"每条记录必须是 mapping，收到 {type(record).__name__}")

    keys = set(record)
    expected = {"chunk_id", "content", "sources", *schema.fields}
    unknown = keys - expected
    if unknown:
        # 包括 created_at / updated_at（模块管理的字段）和拼错的字段名 —— 不静默忽略
        raise ValueError(
            f"记录里有不认识的键 {sorted(unknown)}，本 Schema 只接受 {sorted(expected)}"
        )
    for name in ("chunk_id", "content", "sources"):
        if name not in keys:
            raise ValueError(f"记录缺必需的键 {name!r}")

    chunk_id = record["chunk_id"]
    if not isinstance(chunk_id, str):
        raise TypeError(f"chunk_id 必须是 str，收到 {type(chunk_id).__name__}")
    content = record["content"]
    if not isinstance(content, str):
        raise TypeError(f"content 必须是 str，收到 {type(content).__name__}")

    raw = record["sources"]
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError(f"sources 必须是序列，收到 {type(raw).__name__}")
    if not raw:
        # 没有来源的行没有存在的意义：它是「这个 chunk 来自哪里」的记录
        raise ValueError(f"chunk {chunk_id!r} 的 sources 为空 —— 没有来源就不该入库")
    sources = [_check_source(chunk_id, item) for item in raw]

    for name in schema.fields:
        # 扩展列不写会存成 NULL，但模块要求输入形状统一 —— 这是模块的策略，不是库的要求
        if name not in keys:
            raise ValueError(f"记录缺已声明的字段 {name!r}")
        value = record[name]
        if value is not None and not isinstance(value, str):
            raise TypeError(
                f"字段 {name!r} 的值必须是 str 或 None，收到 {type(value).__name__}"
            )
    return chunk_id, sources


def _check_source(chunk_id: str, item) -> dict:
    """校验一条来源，返回规整后的 `{"path", "start_line", "end_line"}`。"""
    if not isinstance(item, Mapping):
        raise TypeError(f"来源必须是 mapping，收到 {type(item).__name__}")
    expected = ("path", "start_line", "end_line")
    unknown = set(item) - set(expected)
    if unknown:
        raise ValueError(
            f"来源里有不认识的键 {sorted(unknown)}，只接受 {sorted(expected)}"
        )
    for name in expected:
        if name not in item:
            raise ValueError(f"来源缺必需的键 {name!r}（chunk {chunk_id!r}）")

    path = item["path"]
    if not isinstance(path, str):
        raise TypeError(f"path 必须是 str，收到 {type(path).__name__}")
    if not path:
        raise ValueError(f"path 不能为空（chunk {chunk_id!r}）")

    for name in ("start_line", "end_line"):
        value = item[name]
        # bool 是 int 的子类，`True` 混进来会静默存成 1
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} 必须是 int，收到 {type(value).__name__}")
    if item["start_line"] > item["end_line"]:
        raise ValueError(
            f"start_line 不能大于 end_line（chunk {chunk_id!r}："
            f"{item['start_line']} > {item['end_line']}）"
        )
    return {
        "path": path,
        "start_line": item["start_line"],
        "end_line": item["end_line"],
    }
