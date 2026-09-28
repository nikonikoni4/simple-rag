# simple_rag/repository 向量存储模块 —— 设计

> 日期：2026-09-25
> 状态：待评审
> 设计规则见 [CLAUDE.md](../../../CLAUDE.md)
> 探索结论见 [explore/sqlite-vec/FINDINGS.md](../../../explore/sqlite-vec/FINDINGS.md)

## 1. 定位

提供一个独立的向量存储模块。**它不认识任何业务概念** —— 不知道数据是什么、有多少、字段该怎么设计。

**判定标准：如无必要，勿增实体。** 缺了它模块就不能工作的才做；唯一例外是有实测证据说明它会导致错误。

## 2. 两条查询路径

**这是本设计的核心，必须先分清 —— 之前多轮反复的根源就是把它们混成了一条。**

| | **向量相似度查询** `search` | **普通查询** `get` |
|---|---|---|
| 要查询向量 | ✅ 必须 | ❌ 不需要 |
| 要 `k` | ✅ 必须 | ❌ 没有这个概念 |
| 返回排序 | 按距离 | 不按距离 |
| 返回条数 | 最多 k 条 | 取到几行是几行 |
| **普通列** | 能 SELECT、**能过滤** | 能 SELECT、能过滤 |
| **加号列** | 能 SELECT、**不能过滤** | 能 SELECT、能过滤 |

「加号列不能在 KNN 里过滤」这条限制**只在 `search` 里成立**（实测探针 18）。

> 表里 `get` 那列的"能过滤"说的是**普通查询这一类**的固有能力（实测：非 KNN 查询里
> `where content = 'x'` 能跑），**不是 `get` 这个方法的签名** —— `get` 只按 `chunk_id` 取，
> 没有过滤参数。

两条路径**不能互相替代**：用 `search` 代替 `get` 要么逼调用方伪造查询向量，要么因为
「过滤发生在取 Top-K **之前**」而拿不到想要的行（实测：目标组里最好的那行全局排名第 5 时，
`k=5` 先取全局 top5 再筛只剩 1 条，而正确答案是 5 条）。

## 3. 表结构

```sql
create virtual table <内部固定表名> using vec0(
    chunk_id   text primary key,
    embedding  float[{dim}] distance_metric={metric},
    created_at text,
    updated_at text,
    {可过滤字段}  text,      -- 普通列，每字段一行，裸写字段名
    +{其余字段}   text       -- 加号列，每字段一行
);
```

- **每个 `.db` 文件一张表，表名由模块内部固定**
- **DDL 里字段名必须裸写** —— 实测四种引号风格（`"f"` / `'f'` / `[f]` / `` `f` ``）**全部被 vec0 拒绝**（探针 19）
- 空段整段省略（`filterable=()` 时不产生普通列段）；逗号由生成器拼接

### 3.1 固定字段

| 字段 | 谁赋值 | 进 `search` 过滤 |
|---|---|---|
| `chunk_id` | **外部传入** | ✅ 纳入 |
| `created_at` | 模块 | ❌ 不纳入 |
| `updated_at` | 模块 | ❌ 不纳入 |

时间戳格式：**ISO 8601 + UTC**，`datetime.now(timezone.utc).isoformat(timespec="seconds")`。

`insert` 三个都写（`created_at` 与 `updated_at` 同值）；`update` 只刷新 `updated_at`。

> **为什么时间戳放普通列而不是加号列**：它们的值（如 `2025-09-25T10:00:00+00:00`）**超过 12 字符**，
> 真拿去做过滤会踩 §3.3 的悬崖 —— 所以「保留将来按时间筛的余地」这个理由**站不住**。
> 放普通列的真正理由只剩一条：**模块自己要写它们**，放哪类列都行，普通列是短字段的常规位置。
> 不参与过滤是 API 层面的决定。

### 3.2 寻址：`chunk_id` 作主键

**所有操作按 `chunk_id` 寻址，`rowid` 不出现在任何公开签名里。** 实测（探针 17）：

| 拿到什么 | 实测结果 |
|---|---|
| 库强制唯一 | 插入重复值报错（**但异常类型有坑，见 §7.2**） |
| 快速定位 | 按 id 的 delete / update / select 全部 O(1)：50,000 行下 0.012~0.046 ms |
| KNN 照常 | 与无主键表对比：Top50 的 chunk_id 序列与 distance **完全一致** |
| rowid 自然消失 | `select rowid` → `no such column: rowid` |

**为什么不用"模块自己查唯一性"**：`create index` / `create unique index` 在 vec0 上都不支持；
自己 `select ... where chunk_id in (...)` 预查是**全表扫描**（50,000 行 93.3 ms），
而主键冲突检测是**恒定 0.008 ms**。差一万倍以上。

### 3.3 可过滤性由调用方决定 + ⚠️ 12 字符契约

调用方声明的字段，**默认可过滤性关闭**：

| 声明的字段 | 存成哪类列 | 能在 `search` 的 `where` 里用 |
|---|---|---|
| **显式声明可过滤** | 普通列 | ✅ |
| **其余（默认）** | **加号列** | ❌ |

**为什么默认关闭 —— 这是一条实测出来的性能悬崖，不是猜测**（探针 18，5000 行）：

| 过滤字段的**值长度** | 带 `where` 的 KNN | 倍数 |
|---|---|---|
| 1 ~ **12** | 0.15 ~ 0.17 ms | 1.0x |
| **13** | 3.382 ms | **19.9x** |
| 24 | 3.446 ms | 20.3x |
| 500 | 3.991 ms | 23.5x |

**12 是硬边界，不是渐进劣化。** 机制：≤12 字符内联在元数据块里直接比较，>12 字符要**逐行回表**查 `_metadatatext00`。

> ⚠️ 这与「探针 16 说 metadata 列能存长文本且不影响 KNN」**不矛盾** ——
> 探针 16 测的是「长文本**存着不动**」，探针 18 测的是「长值**当过滤条件**」。两件事。

**契约**：`filterable` 声明的字段，**值应当 ≤ 12 字符**。

**模块不校验这条** —— 超过 12 字符的结果仍然**正确**，只是慢。按 CLAUDE.md，慢属于性能
不属"会导致错误"，不该为它加校验。这条写进 `filterable` 的 docstring 与 §12 已知限制。

**文件路径、URL、UUID 这类天生超过 12 字符的值，不适合当可过滤字段** —— 这是这条契约最实际的推论。

### 3.4 列数上限：**两类列各自 16，互不叠加**

实测（探针 18）：

| 普通列 | 加号列 | 结果 |
|---|---|---|
| 16 | 0 | ✅ |
| 17 | 0 | `More than 16 metadata columns were provided` |
| 0 | 17 | `More than 16 auxiliary columns were provided` |
| **16** | **16** | ✅ **可以同时满** |

时间戳占 2 个普通列，所以 **`filterable` 最多 14 个**、**其余字段最多 16 个**。
两条各自超限都在 `create_table` 时抛 `ValueError`。

## 4. 对外 API

```python
from simple_rag.repository import VecDB, Schema, VecSearchResult

with VecDB(db_path) as db:
    # "loc" / "content" 是随手取的字段名 —— 模块不认识它们代表的任何含义
    db.create_table(Schema(
        dim=768,
        metric="cosine",
        fields=("loc", "content"),      # 声明的字段
        filterable=("loc",),            # 从里面挑出哪些能进 search 的 where
    ))

    db.insert([
        {"chunk_id": "c-1", "vector": [...], "loc": "a.md", "content": "..."},
        {"chunk_id": "c-2", "vector": [...], "loc": "a.md", "content": "..."},
    ])

    # 向量相似度查询：找最近的，且限定在 a.md 里
    hits = db.search(query_vector, k=5, where={"loc": "a.md"})

    # 普通查询：按 id 取一行（两类列都能取）
    row = db.get("c-2")

    db.update("c-1", vector=new_vector, fields={"content": "改过的正文"})
    db.delete("c-1")
```

### 4.1 类型

```python
@dataclass(frozen=True)
class Schema:
    dim: int                                       # 1 <= dim <= 8192（实测边界，探针 18）
    metric: Literal["L2", "cosine"]                # 无默认值
    fields: tuple[str, ...] = ()                   # 声明的字段
    filterable: tuple[str, ...] = ()               # fields 的子集

@dataclass(frozen=True)
class VecSearchResult:
    chunk_id: str
    distance: float | None      # search 时是距离；get 时是 None
    created_at: str
    updated_at: str
    fields: dict[str, str | None]
```

- **`metric` 没有默认值**。它在建表时锁死、之后改不了，给默认值等于替调用方拍板。
  （实测：向量归一化后 `L2` 与 `cosine` 的**排序完全相同**，只有距离值不同。）
- **`get` 与 `search` 返回同一个类型**，只是 `distance` 为 `None`。不新增类型。
- **`fields` 的值类型是 `str | None`** —— 实测：普通列**不能为 NULL**，但**加号列可以**
  （不写该字段会读回 `None`）。模块不做非空校验（见 §4.2）。
- **不返回向量** —— 已知代价见 §12。

### 4.2 `insert`

入参是**可迭代的 mapping**，每项一条记录，返回 `None`：

| 键 | 必需 | 说明 |
|---|---|---|
| `chunk_id` | ✅ | 文本，外部传入 |
| `vector` | ✅ | 向量值 |
| 其余键 | — | 必须匹配 `Schema.fields`；缺一个都报错 |

- **方法开头先 `records = list(records)`** —— 入参是可迭代对象，要遍历两遍（校验 + 构造 SQL），
  生成器第二遍会得到空序列、静默插入 0 条
- **整批一个事务**：全部成功或全部失败
- **能提前校验的都在开事务之前对整批做完**
- **`chunk_id` 冲突无法提前查**（批内重复只有库知道）。这类失败发生在事务中途，
  **由模块显式 `rollback` 保证不留半批数据** —— 实测这不是引擎自动保证的（探针 18 §8.5）
- 自动填 `created_at` 与 `updated_at`

逐条校验规则：

| 情况 | 结果 |
|---|---|
| 缺 `chunk_id` / `vector` | `ValueError` |
| 出现 `Schema.fields` 之外的键（含拼错的字段名）| `ValueError` —— **不静默忽略** |
| 出现 `created_at` / `updated_at`（模块管理的字段）| `ValueError` |
| 缺某个已声明的字段 | `ValueError` —— **模块自己拦，不依赖库**。理由：普通列不能为 NULL 而**加号列可以**，两类行为不同，靠库只能拦住一半 |
| 字段值不是 `str` / `None`（如 `123` / `b"x"`）| `TypeError` |
| `chunk_id` 不是 `str` | `TypeError` |
| 传入的不是 mapping | `TypeError` |
| 空输入 | 直接返回，不报错 |
| `chunk_id` 重复（批内或与库里已有冲突）| `ValueError` —— 库抛 `OperationalError`，模块字符串匹配后翻译（见 §7.2）|
| 向量无法转成 float32 数组（如 `{"a":1}`）| `TypeError`（numpy 直接抛，**模块不包装**）|
| 向量维度 / NaN / 零向量不合法 | `ValueError`（见 §6）|

> 加号列不写会读回 `None`，这是**允许**的 —— 所以"缺某个已声明字段"报错是模块的策略
> （保证 `insert` 的输入形状统一），不是库的要求。

### 4.3 `search`

```python
db.search(vector, k, where=None) -> list[VecSearchResult]
```

- **`k` 必填**（理由同 `metric`），必须是 `int`（排除 `bool`），`0 <= k <= 4096`
  （实测边界，探针 18：`4097` → `k value in knn query too large ... the limit is 4096`）
- 用 `k = ?` 约束，不用 `order by distance limit`
- **距离并列时的顺序不是稳定契约**
- 返回条数可能少于 `k`

### 4.4 `where`

只作用于 `search`，**只支持等值**：

```python
{"loc": "a.md"}                       # 可过滤的声明字段
{"chunk_id": "c-1"}                   # 固定字段 chunk_id 也允许
{"loc": "a.md", "chunk_id": "c-2"}    # 多键 = AND（实测可用）
```

- key 必须是**可过滤的字段名**或 `chunk_id`，否则 `ValueError`
- value 必须是 `str`，否则 `TypeError`
- `None` 或空 dict = 不过滤

### 4.5 `get`

```python
db.get(chunk_id) -> VecSearchResult | None      # 不存在返回 None
```

- `chunk_id` 必须是 `str`，否则 `TypeError`
- **不受 `k`、不受距离、不受列类型限制** —— 普通查询，两类列都能取
- **没有过滤参数**（要过滤用 `search` 的 `where`）

### 4.6 `update`

```python
db.update(chunk_id, vector=None, fields=None) -> int    # 返回受影响行数
```

- `chunk_id` 必须是 `str`，否则 `TypeError`
- 返回 `0` 表示 chunk_id 不存在，不报错
- `fields` 是**局部更新**：只改给出的键；key 必须是声明的字段名，value 是 `str` 或 `None`
- `fields={}`（空 dict）→ `ValueError` —— 实测空 SET 子句会让 SQLite 报
  `near "where": syntax error`（探针 18 复现）
- `vector` 与 `fields` 都为 `None` → `ValueError`
- **`vector` 必须走同一个 `_codec.encode`**
- 自动刷新 `updated_at`
- **绝不生成 `SET chunk_id = ...`** —— 实测库会报 `UPDATEs on vec0 primary key values are not allowed.`

> ⚠️ **`update` 是直接改列，不是「先删再插」。** 实测 vec0 **支持**改向量列和各列（含加号列）。
> 走 delete+insert 会**重置 `created_at`**，违反 §3.1。

### 4.7 `delete`

```python
db.delete(chunk_id) -> int      # 返回删除行数，不存在返回 0
```

- `chunk_id` 必须是 `str`，否则 `TypeError`

### 4.8 `close`

关闭连接并置空**连接**（Schema 保留）。支持 `with` 语句。重复调用是 no-op。
关闭后再调用其他方法**不做额外检查** —— sqlite3 自己会抛 `ProgrammingError`。

## 5. 字段名校验

DDL 是字符串拼接，而 **vec0 在 DDL 里拒绝任何带引号的标识符**（实测四种风格全被拒，探针 19），
**所以字段名必须自己校验**。

| 规则 | 依据 |
|---|---|
| 匹配 `^[A-Za-z][A-Za-z0-9_]*$` | **实测**：前导下划线被拒（`_loc` → `Could not parse`）；含空格逗号会让 DDL **静默拆成多列**（`body text, extra` → 真建出 `body` 与 `extra`）；带引号被拒 |
| 不能重复（**大小写不敏感**）| **实测**：`loc` + `LOC` → `duplicate column name: LOC` |
| 不能用保留名 | `embedding` / `chunk_id` / `distance` / `k` 是**库级拒绝**（实测 `duplicate column name`）；**`created_at` / `updated_at` / `vector` 是模块策略**（库其实允许这些列名，是模块自己要用）|
| `filterable` 必须是 `fields` 的子集 | 模块规则 |
| `filterable` 的元素也要过同一个正则 | 模块规则 |
| `dim` 在 `1..8192` | **实测**（探针 18）|
| `filterable` ≤ 14 个、其余字段 ≤ 16 个 | **实测**（探针 18，两类列各 16 上限）|

违规一律 `ValueError`。

### 5.1 模块在 DML 里给标识符加方括号

**SQL 关键字不需要进黑名单** —— 那要列全 SQLite 的上百个关键字，列不全就漏。
实测（探针 19）找到更干净的办法：

```sql
-- DDL：裸写（vec0 要求）
create virtual table v using vec0(chunk_id text primary key, embedding float[768], from text);

-- DML：加方括号（避开关键字，且拼错会报错）
select [chunk_id], [from] from v where embedding match ? and [from] = ? and k = 5;
```

| 风格 | DML 能用 | 拼错列名时 |
|---|---|---|
| **方括号** | ✅ 全部 | ✅ `no such column` |
| 反引号 | ✅ 全部 | ✅ `no such column` |
| 双引号 | ✅ 全部 | ❌ **静默返回字符串字面量** |
| 裸写 | ❌ 关键字崩 | ✅ `no such column` |

⚠️ **不要用双引号**：SQLite 的双引号在找不到列名时会**退化成字符串字面量** ——
拼错列名不报错，而是静默拿到一个常量。正是本项目专门要防的那类静默错误。

**两条规则不同，必须分开：DDL 裸写、DML 加方括号。**
（加号列在 DML 里是**去掉 `+` 的普通名字**，加方括号同样有效，已验证。）

## 6. 向量编码（`_codec`）

```python
def encode(vec, dim: int) -> bytes:
    arr = np.asarray(vec, dtype=np.float32)      # 统一转换，不依赖库校验
    if arr.ndim != 1:
        raise ValueError(...)
    if arr.shape[0] != dim:
        raise ValueError(...)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(arr).all() or norm == 0.0:
        raise ValueError(...)                     # NaN / Inf / 零向量
    return (arr / norm).astype(np.float32).tobytes()
```

- **`int32` / `float64` 是被接受并转换的**，不是"非法 dtype"（实测探针 09）
- 无法转换的输入由 numpy 自己抛（`{"a":1}` → `TypeError`，`["a","b"]` → `ValueError`），
  落在 §7 异常契约的同一格子里，**模块不做包装**
- **归一化无条件执行，无开关**

**为什么必须做**：

| 坑 | 实测后果 | 出处 |
|---|---|---|
| dtype 写错 | `float64` 混进 `float[8]` 列 → **静默存成垃圾向量** | 探针 09 |
| NaN / Inf | 被接受，且污染 KNN 排序（NaN 行排最前）| 探针 09 |
| 零向量 | `0/0` 归一化后变成 NaN —— 会绕过上面的 NaN 检查，所以必须在除法**之前**显式拦 | **数学推导，不是实测** |

**归一化对查询向量同样施加。** 实测查询端归一化对**排序**是 no-op（探针 15），
统一施加是为了让两条路径共用同一个 `encode`、避免"两端不一致"这类隐性假设；
它对 **L2 的距离数值**有影响（cosine 不受影响，因其尺度不变）。

## 7. 异常契约

| 类型 | 含义 |
|---|---|
| `ValueError` | 输入不合法（值不对、范围不对、chunk_id 冲突）|
| `TypeError` | 类型不对 |
| `sqlite3.Error` | 库层面的问题 |

### 7.1 为什么不做错误翻译层

sqlite-vec 有一句极具误导性的报错，实测它会被多种**完全不同**的问题触发：

```
A LIMIT or 'k = ?' constraint is required
```

| 触发条件 | 本设计 |
|---|---|
| partition key 上 `IN` / `OR` | 不支持 partition key |
| 辅助列出现在 WHERE | 模块只把可过滤字段和 `chunk_id` 放进 `where`（§4.4 的 key 校验），加号列进不了 SQL |
| 不写 `k` / `LIMIT` | `k` 是必填参数，恒用 `k = ?` |
| KNN + JOIN + `order by distance limit` | 模块不拼 JOIN |

**这些在本设计里都构造不出来，所以不需要翻译层。**

### 7.2 但 `chunk_id` 冲突是个例外，必须翻译 ⚠️

**实测：vec0 抛的是 `OperationalError`，不是 `IntegrityError`**（探针 18 §8.4）。

```python
sqlite3.OperationalError: UNIQUE constraint failed on t primary key
    isinstance(e, sqlite3.IntegrityError)  ->  False
    sqlite_errorcode                        ->  1        # SQLITE_ERROR
    sqlite_errorname                        ->  'SQLITE_ERROR'

# 普通 SQLite 表（对照）
sqlite3.IntegrityError: UNIQUE constraint failed: plain.id
    sqlite_errorcode                        ->  1555
```

**后果**：

1. 按常规写 `except sqlite3.IntegrityError` —— **永远不触发**，§7 的契约静默失效
2. **用普通 SQLite 表写的单测会全绿** —— 只有真跑 vec0 才暴露
3. `sqlite_errorcode` 是通用的 `1`、**不能用来判别**，唯一可行的是匹配报错文本

**所以必须做一次字符串匹配**，把 `UNIQUE constraint failed on <表名> primary key` 翻成 `ValueError`。
这是全文**唯一**一处错误翻译，理由不是"报错不够友好"，而是
**`OperationalError` 同时也是"库真的坏了"那一类，不翻就无法区分**。

## 8. 连接与事务

### 8.1 连接

```python
conn = sqlite3.connect(db_path)
conn.enable_load_extension(True)
sqlite_vec.load(conn)
conn.enable_load_extension(False)      # 必须立刻关
conn.execute("pragma journal_mode=WAL")
```

- **`enable_load_extension(False)` 必须紧跟 `load()`** —— 实测关掉后再 `load()` 报 `not authorized`
- **连接由 `VecDB` 实例持有，不是模块级全局**。按需创建，可同时存在多个。
  **模块 import 时不建立任何连接**
- **不加 `check_same_thread=False`**，保持 Python 默认的线程亲和性
- 构造失败时先 `close()` 再抛 —— 用局部变量 + 判空，否则 `connect` 本身失败时 `close()` 会抛
  `AttributeError` 盖掉真实错误
- **不关连接，`.db` 文件删不掉**（实测 `WinError 32`）；`close()` 之后立刻可删

### 8.2 事务

**必须显式控制。** Python 的 `sqlite3` 默认 `isolation_level=''`，会在 DML 前**隐式** `BEGIN`；
此时模块再发 `BEGIN` 会报 `cannot start a transaction within a transaction`。

**做法**：构造时设 `conn.isolation_level = None`，模块自己发 `BEGIN` / `COMMIT` / `ROLLBACK`。

**`rollback` 必须显式调用。** 实测（探针 18 §8.5）：冲突后 `conn.in_transaction` 仍为 `True`，
失败语句不会自动中止事务 —— **若 except 分支里调的是 `commit()`，半批数据会真的落库**。

## 9. 建表与重开

### 9.1 `create_table(schema)` 的语义

```python
def create_table(schema):
    # 表不存在      -> 建表，把 schema 存进实例
    # 表已存在      -> 读 sqlite_master.sql，比对 dim 与 metric
    #                  一致   -> 直接返回
    #                  不一致 -> ValueError（"打开的不是这个 Schema 建的库"）
    # 无论哪条路径，都把本次传入的 schema 存进实例
```

**为什么必须定义这个**：`VecDB(path)` 之后无条件调 `create_table` 是**最常规的用法**
（每次开库都调）。不定语义，第二次打开同一个 `.db` 就会 `table already exists` 报错。
用 `create table if not exists` 只能避免报错，**但会静默接受一个结构不同的库**。

**比对 dim 与 metric 怎么读回来**（实测探针 18 §8.8）：

```sql
select sql from sqlite_master where name = '<表名>' and type = 'table'
```
```
CREATE VIRTUAL TABLE v using vec0(chunk_id text primary key, embedding float[32]
  distance_metric=cosine, created_at text, updated_at text, loc text, +content text)
```

dim、metric、主键、哪些是加号列**全在里面**，而且**不带扩展的连接也能读**。

**只比 dim 与 metric，不做全 DDL 比对。** 全字符串比对会在模块自己的 DDL 生成器改动时误报；
而 dim / metric 是**唯一两个会静默出错**的参数（其余字段不符会在 insert/search 时明确报错）。

> 我之前在 spec 里写过「Schema 不可能从库里读回来」—— **那句是错的**，理由是
> `pragma table_info` 的类型列是空字符串。它不给类型，但结构在 `sqlite_master.sql` 里。

### 9.2 重建的路径

vec0 不能 `ALTER`，所以改 dim / metric 只能重建。**模块不提供 drop / 重建入口**，
调用方：`close()` → 删 `.db`（连同 `-wal` / `-shm`）→ 新建实例 → `create_table`。

## 10. 范围

**做**

| 功能 | 理由 |
|---|---|
| 连接 / 关闭 | 不加载扩展表建不了；不关连接 Windows 上 `.db` 删不掉 |
| 建表（含重开校验）| 不建表存不进去；不定重开语义则无法跨进程复用 |
| 插入 | — |
| `search`（向量相似度）| 模块的存在理由 |
| `get`（按 id 取一行）| 普通查询，`search` 无法替代（见 §2）|
| 删除 | vec0 无 UPSERT。**注意这不是"更新的前置"** —— `update` 直接改列（§4.6）；只有**块集合变化**（增块/删块）时才需要 delete + insert |
| 更新 | 模块需要能改已有数据 |
| **归一化 / dtype 转换** | 实测不做会静默损坏数据（见 §6）|
| **DML 标识符加方括号** | 实测不加则字段名碰 SQL 关键字时永远插不进数据（见 §5.1）|

**不做**

| 砍掉 | 为什么 |
|---|---|
| partition key | 它是性能优化（分片加速），按 CLAUDE.md 属下游决定 |
| 字段类型 int/float | 只支持文本也能工作；数字列行为未实测 |
| `L1` 度量 | 未实测，且与"始终归一化"冲突 |
| 范围 / `IN` / `OR` 过滤 | 只支持等值也能工作；当前无需求 |
| 版本号 / 迁移机制 | 不做也能工作 —— §9.1 的 dim/metric 校验已经覆盖了"打开错的库"这个真正会静默出错的场景 |
| `get` 返回向量 | 不返回也能工作。已知代价见 §12 |
| 自定义异常类 | 内置的 `ValueError` / `TypeError` / `sqlite3.Error` 已够（见 §7）|
| 批量 `delete` / 批量 `update` / `insert` 返回值 | 单条已能工作。**已知代价**：逐条 delete/update 是 N 个事务，没有跨记录的原子性（见 §12）|
| `close()` 后的状态检查 | sqlite3 自己会抛 `ProgrammingError` |
| drop / 重建入口 | 调用方 close 后删文件即可（§9.2）|

## 11. 文件布局与测试

```
simple_rag/repository/
  __init__.py      对外出口：VecDB / Schema / VecSearchResult
  vec_db.py        VecDB 类（持有连接 + Schema + 事务控制）
  _connection.py   建连接（加载扩展）+ 关闭
  _schema.py       Schema 校验 + DDL 生成（纯函数）
  _codec.py        归一化 + dtype（纯函数）
  _store.py        SQL 文本构造（纯函数，返回 (sql, params)）
```

**Schema 由 `VecDB` 实例持有**，`create_table` 时存入。未调 `create_table` 就调其他方法 → `ValueError`。

**校验职责划分**（避免散落或重复）：

| 谁 | 校验什么 |
|---|---|
| `_schema` | **Schema 本身** —— 字段名规则、保留名、`filterable ⊆ fields`、`dim` 边界、两类列数上限 |
| `_store` | **一条记录 / 一个 `where` / 一个 `k` 相对 Schema 是否合法** —— 纯函数 |
| `_codec` | **向量本身** —— 维度、dtype、NaN、零向量 |
| `vec_db` | 类型级兜底、事务控制、`chunk_id` 冲突翻译（§7.2）、重开时的 dim/metric 比对（§9.1）|

| 组件 | 测试方式 | 覆盖重点 |
|---|---|---|
| `_codec` | **纯单测** | `float64`/`int32` 应**转换**；`{"a":1}` 应 `TypeError`；`["a"]`/`None` 应 `ValueError`；NaN；**零向量**；维度不符；归一化结果 |
| `_schema` | **纯单测** | 字段名规则（含空格逗号、前导下划线、大小写重复、保留名）、`filterable ⊆ fields`、`dim` 边界 0/1/8192/8193、两类列数上限 14/16、DDL 生成（含空 filterable 段） |
| `_store` | **纯单测** | `where` 子句构造（方括号风格）、多键 AND、非法 key 被拦、`k` 的 bool/范围校验、空 `fields` 被拦 |
| 端到端 | 临时库（tempfile）| 建表→插入→search→get→update→delete；**重开已有库**；**dim/metric 不符时报错**；`chunk_id` 唯一性由库强制且**翻译成 ValueError**；**SQL 关键字字段名能正常增删改查**；整批失败回滚；时间戳自动维护；`close()` 后可删文件 |

## 12. 已知限制

- **KNN 是暴力全表精算，不是 ANN**（实测：耗时随行数近似线性；但只在 128 维假向量上测到 50,000 行）
- **度量与维度建表时锁死**；重开时靠 §9.1 的校验发现不符
- **vec0 无 UPSERT**：插入已存在的 `chunk_id` 会报错，模块翻成 `ValueError`
- **普通列不能为 NULL，加号列可以**（不写会读回 `None`）
- **列数上限**：普通列 16（含两个时间戳）、加号列 16
- **⚠️ 可过滤字段的值超过 12 字符，带 `where` 的 KNN 慢约 20 倍**（§3.3）。模块不拦，调用方负责
- **删除不回收存储空间**；全量重建（删全部再插回）**每轮泄漏约一个 chunk**，`VACUUM` 止不住。
  要重建就删库重建
- **「改文档 = delete + insert 同 chunk_id」不泄漏**（实测探针 20：500 篇各改 5 轮，
  chunks 恒为 1、页数恒 57、文件恒 4.0 KB）—— 日常重索引安全，不需要删库重建
- **没有跨记录的事务边界**：`insert` 整批一个事务，但 `delete` / `update` 是单条单事务。
  「一篇文档 20 个块全换掉」= N 次调用、N 个事务，中途失败会留下半新半旧。
  **这是主动取舍**（批量入口是按"如无必要勿增实体"砍掉的），需要时再加
- **单条 `update`/`delete` 在磁盘库上约 2.1 ms**（实测探针 20，文件库 + WAL）：
  `update` 向量或普通列 2.130 ms、`delete`+`insert` 4.450 ms、**对照普通 SQLite 表只要 0.002 ms**。
  是 vec0 写路径的持久化开销。**同一事务内批量做会摊平：2000 条向量更新，
  逐条 5.50 s vs 整批 0.045 s，差 122 倍。**
  （注意：**内存库里同样的操作只要 0.026 ms** —— WAL 在内存库是空操作，别拿内存库测写开销）
- **单表**：一个 `.db` 文件一张表
- **一个实例不应跨线程使用**
- **多实例写同一个 `.db`**：第二个会等到 sqlite3 默认的 5 秒超时后抛 `database is locked`
- **`k` 上限 4096**，且没有分页
- **`get` 不返回向量** —— 所以「加字段要重建表」时读不回旧向量，只能重跑 embedding
- **真实 embedding 端到端未验证**：探针只用了假向量
- **打包后的实际运行未验证**：PyInstaller 必须显式收集 `vec0.dll` 且**不能改名**
  （见 [explore/pyinstaller/FINDINGS.md](../../../explore/pyinstaller/FINDINGS.md)）
