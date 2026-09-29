---
version: 1.0
created_at: 2026-09-29
updated_at: 2026-09-29
last_updated: 创建索引，登记 BM25 的 k1 / b 不可调整一条
abstract: docs/known-limitations 目录索引。记录当前系统有意接受、且依赖特定前提才成立的约束，供 AI 与开发者判断「这里不能动 / 这里这样是有原因的」。
---

## 2026-09-29-fts5参数不可调

- updated_at: 2026-09-29
- path: `docs/known-limitations/2026-09-29-fts5参数不可调.md`
- 触发规则：当涉及 BM25 的打分参数 `k1` / `b`，或准备为检索质量调参时阅读
- 内容摘要：BM25 的打分由 SQLite 内置 FTS5 的 `bm25()` 完成，`k1` / `b` 被 FTS5 硬编码在 1.2 / 0.75，SQL 层没有设置入口；给 `bm25()` 传额外参数（`bm25(t, 1.2, 0.75)`）是**列权重**而非 `k1` / `b`，调不到。严重程度低，默认参数够用。
