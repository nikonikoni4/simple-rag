---
version: 1.1
created_at: 2026-09-29
updated_at: 2026-09-29
last_updated: fts5 参数不可调一条更新为 v1.1 —— 已可换 rank_bm25 实现绕过
abstract: docs/known-limitations 目录索引。记录当前系统有意接受、且依赖特定前提才成立的约束，供 AI 与开发者判断「这里不能动 / 这里这样是有原因的」。
---

## 2026-09-29-fts5参数不可调

- updated_at: 2026-09-29（v1.1）
- path: `docs/known-limitations/2026-09-29-fts5参数不可调.md`
- 触发规则：当涉及 BM25 的打分参数 `k1` / `b`，或准备为检索质量调参时阅读
- 内容摘要：BM25 的 **fts5 实现**打分由 SQLite 内置 FTS5 的 `bm25()` 完成，`k1` / `b` 被硬编码在 1.2 / 0.75，SQL 层没有设置入口；给 `bm25()` 传额外参数是**列权重**，调不到。**已可绕过**：`create_bm25("rank_bm25", k1=..., b=...)` 可调参（代价是无增量，只能 rebuild）；FTS5 本身的限制仍在，fts5 传 k1/b 会显式报错。
