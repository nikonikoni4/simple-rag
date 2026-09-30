# simple_rag/chunking/structured_file —— Markdown 按标题切分

把一个 Markdown 文件切成可入库的 chunk。**只认结构（标题 / 代码块 / 表格），
不认识任何业务概念** —— 什么算一篇文档、有哪些字段、要不要按条件筛选，
都是调用方的决定。

- 设计规则：[CLAUDE.md](../../../CLAUDE.md)
- 端到端实测（真实语料 + 专门造的代码块 / 表格语料）：
  [explore/实际测试/向量嵌入测试](../../../explore/实际测试/向量嵌入测试/README.md)

## 用法

```python
from pathlib import Path
from simple_rag.chunking.structured_file.md_chunk_by_title import chunk_by_title
from simple_rag.chunking.structured_file.types import render_text

drafts = chunk_by_title(
    Path("diary/2025-01-16.md"),
    max_token=1024,
    start_line=0,
    summary_func=my_summarize,   # 可选；模块不绑定任何模型服务，传 None 就整步跳过
    min_token=30,                # 可选；短块阈值，None = 不合并。见「短块合并」
)

text = render_text(drafts[0].segments, with_summary=True)   # 送进 embedding 的文本
```

## 流程（5 步）

| 步 | 做什么 | 产物 |
|---|---|---|
| 1 | 读文件 + 按 `start_line` / `end_line` 切片 | 文本 |
| 2 | 清洗：去行尾空格、连续空行收敛成一行 | 文本（**行数会变少**） |
| 3 | 单遍扫描 + 栈建标题树，后序回填 token 统计 | `MDFileHead` 树 |
| 4 | 识别代码块 / 表格挂到 `special_content`；可选**并发**总结 | 树（带块与摘要） |
| 5 | 切分 | `list[ChunkDraft]` |

第 4 步的「识别」与「总结」是分开的：识别**总要跑**（`special_content` 是事实层，
切分要靠它保护块、把摘要带进 chunk），总结只在给了 `summary_func` 时才跑。

## 切分策略

**装箱** —— 尽量保持同一个节点的内容在同一个 chunk 内：

1. 整棵子树装得下 `max_token` → 整块并入当前 chunk
2. 装不下 → 先放本节点正文，再按阅读顺序逐个并入子节点
3. 加不下 → 结算当前 chunk，从下一个子节点重开
4. 正文自己就超限 → 切开

**切点**：

- 默认落在**行边界** —— 跟作者怎么换行一致
- **一行自己就超过 `max_token`** 时改在这一行**内部**切：取预算内**最靠后**的
  句末标点（`。！？；!?;`）之后切开，每片尽量接近上限又不劈断句子；
  一个可用标点都没有就**按字符硬切**
- 代码块 / 表格是**硬边界**，不会从中间切开（切点落进去就回退到块首）

**重叠** —— 相邻 chunk 之间留 `max_token × 15%` 的重叠区，跨切点的内容上下两块
都能召回。切点是被特殊块顶回来的话**不留重叠**：否则下半段会从重叠处再切一次、
又退到同一个切点，切出一块被上一块完全包住的重复 chunk。

**短块合并** —— `min_token` 非 `None` 时，`tokens` 小于它的 chunk 会被并进相邻
chunk：优先并进**上一个**（短块多半是紧跟大块之后被 flush 出来的），没有上一个就
并进**下一个**（文档开头就是短块时），两边都没有就**丢弃**（整篇只切出一个短块）。
合并只拼相邻块，**不重新装箱**，所以合并后的 chunk 可能超过 `max_token` —— 这是
刻意的取舍，短块单独成块对检索的伤害比这一点溢出大。

短 chunk 在检索里会当「吸引子」：文本越短，向量越「通用」，跟什么查询都不算远。
实测一个 13 token 的块在多个不相关查询里排到第 1，把正确答案挤到第 2。
**`min_token` 不给默认值** —— 取多少得按自己的语料实测，见下。

### 实测：`data/` 全部语料（385 个文件，`max_token=1024`）

| `min_token` | chunk 数 | 剩余 `<30` | 超 `1024` | 返回空列表的文件 |
|---|---|---|---|---|
| 不合并 | 661 | 134（20.3%） | 2 | 72 |
| 30 | 527 | 0 | 3 | 195 |
| 100 | 509 | 0 | 7 | 206 |
| 300 | 447 | 0 | 17 | 254 |

读法：

- **短块不是切分碎片**，是**日记文件的头部**（日期行 + H1 标题行）和孤立的 `#` 行 ——
  `min_token=30` 时 134 个短块全部消化掉，chunk 数降 20%
- 合并会让**超限变多**（2 → 3 → 17）：短块是塞进已经装满的邻居里的
- **返回空列表的文件从 72 涨到 195**。多出来的 123 个全是 9~71 字节的「空壳日记」
  （只有日期和标题、零正文），丢弃**不损失任何内容**，已在语料上逐个核对
- 阈值再往上（100 / 300）收益递减、超限代价加速上涨

## 类型之间的关系

```
MDFileHead   标题树节点（中间结构，只承载结构与 token 统计）
    ↓ 切分
Segment      一段：面包屑 + 纯正文 + 行区间 + 落在本段内的特殊块
    ↓ 装箱
ChunkDraft   一个 chunk：若干 Segment（还没有向量）
    ↓ embedding
Chunk        最终产物，带向量，可直接交给 VecDB
```

**面包屑和摘要都不写进 `Segment.text`。** 存的时候分开存，用的时候才拼 ——
给不给 agent 摘要、给不给面包屑，是调用方的决定，不该在切分这一步就合成一段
还原不回去的文本：

```python
render_text(draft.segments)                       # 面包屑 + 正文
render_text(draft.segments, with_summary=True)    # 再把摘要插在各自的块之前
draft.segments[0].pref                            # 单独拿面包屑
draft.special_content[0].summary                  # 单独拿摘要
```

摘要插在它那块**之前**、原块原样保留，定位靠**行号**而不是搜文本 ——
正文里出现两张一样的表格时，搜索会定位到错的那一张。

## 调用方要知道的约束

| | |
|---|---|
| **`max_token` 是软约束** | 两种突破：① 原子块自己就超预算 —— 整块保留。纯文本本身不会超限 ② 短块合并把内容塞进已经装满的邻居 |
| **`max_token` 不含摘要** | 送进 embedding 的文本 = 面包屑 + 正文 + 本 chunk 内所有摘要。设 `max_token` 时自行留余量 |
| **`min_token` 不给默认值** | 短块阈值由调用方按自己的语料实测决定，模块只给开关。`None` = 不合并，短块会留在结果里 |
| **可能返回空列表** | 给了 `min_token` 且整篇只切出一个短块时。语料里的「空壳日记」会走这条路 —— 这是刻意的 |
| **摘要是可选、且会软失败** | `summary_func` 由调用方注入。单个块抛异常只记 `logging.warning`，该块 `summary` 留 `None`，不中断其余块 |
| **`chunk_id` 按内容寻址** | `H(file_path + content_hash)`，**行号不参与**。内容一样就是同一条记录；内容变了主键就变，旧行要调用方按 `file_path` 清掉 |
| ⚠️ **入库前必须按 `chunk_id` 去重** | 一批内 + 库里已有的都要。同 id 撞的是 `UNIQUE`，而 vec0 抛 `OperationalError`、`sqlite_errorcode` 是通用的 `1`，只能字符串匹配 |
| **`file_path` 原样保留传入值** | 它直接参与主键。传相对路径（相对语料根）才能跨机器可移植；传绝对路径会把主键绑死在这台机器上 |
| **行号相对清洗后文本** | `_clean_file_content` 会吃掉连续空行，所以不等于原文件行号，映射不是简单加偏移 |
| **行号只是参考信息** | 同一行切出的多段，`start_line` / `end_line` 完全相同。别拿它当唯一标识 |

## 已知限制

1. **代码围栏里的 `# 注释` 会被误判成标题**；Setext 式标题（下一行跟 `===`）不识别
2. **标题跳级**（`#` 直接到 `###`）不报错，按「就近挂到栈顶父节点」处理
3. **摘要变了 `content_hash` / `chunk_id` 都不变** —— 要不要重嵌得调用方自己判断
4. **默认不合并短块** —— 不传 `min_token` 时短块照样单独成 chunk，会在检索里当
   「吸引子」。传 `min_token` 可以消除，但阈值得自己实测（见「短块合并」）

## 文件

| 文件 | 职责 |
|---|---|
| `types.py` | 跨步骤类型：`MDFileHead` / `SpecialContent` / `Segment` / `ChunkDraft` / `Chunk`，以及拼装用的 `render_text` |
| `md_chunk_by_title.py` | 全部实现：清洗、建树、识别特殊块、并发总结、切分 |
| `__init__.py` | 空 —— 入口直接从 `md_chunk_by_title` 取 |

测试：`tests/chunking/structured_file/test_md_cut.py`、`test_md_summary.py`、
`test_segment_render.py`（`python -m pytest tests/chunking/structured_file`）。
