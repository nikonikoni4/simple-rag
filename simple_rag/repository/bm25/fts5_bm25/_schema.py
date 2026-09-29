"""FTS5 表的形状与 DDL 校验（纯函数）。

**表结构完全固定** —— 不像 `vec` 那样由调用方给 `Schema`（那里维度、度量、
字段名都要选）。这里只有一张两列的表，没有可配置项。

唯一选过的是 `tokenize`：固定 `unicode61`，因为**中文分词在 Python 侧做**
（`simple_rag.tokenization`），FTS5 只负责按空格切开已经分好的 token 串。

实测（[explore/bm25/FINDINGS.md](../../../../explore/bm25/FINDINGS.md) §4.1）：
`unicode61` 直接吃中文会把**整句**当成一个 token，查 `检索` 零命中。
"""

from __future__ import annotations

FTS_TABLE_NAME = "chunks_fts"
TOKENS_COLUMN = "tokens"
CHUNK_ID_COLUMN = "chunk_id"
FTS_TOKENIZER = "unicode61"


def build_ddl() -> str:
    """生成建表语句。

    三段都不能少：

    - `tokens` —— 存 token 串（空格分隔），倒排索引建在它上面
    - `chunk_id unindexed` —— 存原始 id，**只存不索引**（它是输出用的关联键，
      不参与 `match`）
    - `tokenize='unicode61'` —— 按空格切，不做中文分词
    """
    return (
        f"create virtual table {FTS_TABLE_NAME} using fts5("
        f"{TOKENS_COLUMN}, "
        f"{CHUNK_ID_COLUMN} unindexed, "
        f"tokenize='{FTS_TOKENIZER}'"
        f")"
    )


def verify_ddl(sql: str) -> None:
    """已有的 `chunks_fts` 必须与本模块建的一致，否则 `ValueError`。

    **必须拦。** tokenizer 不同 = token 不同 = **静默零召回、不报错** —— 换成
    `trigram` 之后中文 2 字词全部查不到，而库不会报任何错
    （实测 [FINDINGS.md](../../../../explore/bm25/FINDINGS.md) §4.2）。

    对应 `vec` 那边「不能用 `if not exists` 蒙混过去」的同一条规则。
    """
    normalized = " ".join(sql.lower().split())
    problems: list[str] = []
    if "using fts5" not in normalized:
        problems.append("不是 FTS5 虚拟表")
    if TOKENS_COLUMN not in normalized:
        problems.append(f"没有 {TOKENS_COLUMN!r} 列")
    if f"{CHUNK_ID_COLUMN} unindexed" not in normalized:
        problems.append(f"没有 {CHUNK_ID_COLUMN!r} unindexed 列")
    if FTS_TOKENIZER not in normalized:
        problems.append(f"tokenizer 不是 {FTS_TOKENIZER!r}")
    if problems:
        raise ValueError(
            f"这个 {FTS_TABLE_NAME!r} 不是本模块建的：{'；'.join(problems)}。"
            f"库里是 {sql!r}"
        )
