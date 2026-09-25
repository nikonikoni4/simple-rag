# RAG

这是一个学习项目，结合 `D:\desktop\quackDocs\my_notes\知识库\raw\personal-notes\编程\RAG` 一边学习一边进行。

## 演进路线

这个仓库是**学习用的单体实现**，不是最终形态。

- **当前**：在 `simple_rag/` 里直接写实现。优先跑通和看懂，不追求抽象。
- **之后**：进展到一定程度后，会把它**重写并抽象为一个独立的 SDK 包**，再引入到桌面项目
  **LifePrism** 中使用。

顺序是有意的：先有能跑的具体实现，再从真实需求里提炼接口 —— 而不是一开始就为了"通用"设计出用不上的抽象。

设计规则见 [CLAUDE.md](CLAUDE.md)。

## 目录

| 目录 | 说明 |
|---|---|
| `simple_rag/` | 正式实现 |
| `explore/` | 一次性验证代码（探针）与结论。**不是正式实现，别拿去复用** |
| `data/` | 语料（LifePrism 日记数据） |

# 技术选型

## 1. 向量数据库：sqlite-vec

为什么：LifePrism 使用的是 sqlite，而且数据量并不算大，不需要 ANN；而且 RAG 并不是当前项目的主要功能，作为桌面应用扩展，相对于其他数据库需要开销小。

**选型已用探针验证**：见 [`explore/index.md`](explore/index.md)。

结论是**可用**，但有三个必须在实现里主动防的坑：

1. **dtype 写错会静默损坏数据** —— `np.float64` / `np.int32` 的字节都会被照单全收，不报错
2. **partition key 用不了 `IN` / `OR`，也不能 UPDATE** —— 且报错信息极具误导性
3. **PyInstaller 必须显式收集 `vec0.dll`** —— LifePrism 现在的 `lifeprism.spec` 里没有

## 2.

待补充。
