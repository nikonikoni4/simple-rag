"""rank_bm25 实现的 pickle 持久化（纯函数，不碰索引状态）。

**存纯数据，不存 `BM25Okapi` 对象** —— 对象 pickle 绑定库的内部属性布局
（`doc_freqs` / `idf` / `doc_len` ...），库一升级改属性名，旧文件全部报废。
纯数据（chunk_id + 分好的 token）只绑定我们自己的稳定形状，加载后重算
一遍 BM25 统计（纯内存计数，快）。

⚠️ pickle 反序列化可执行任意代码 —— 本模块的文件只应该来自**自己写的**
`save`，不要加载来路不明的 `.pkl`。
"""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path

# 数据格式版本。改了 `records` 的形状就 +1，`load` 里拒载旧文件 ——
# 比「unpickle 出错或静默错位」清楚得多。
_FORMAT_VERSION = 1

# records: [(chunk_id, tokens), ...] —— tokens 是分好的 token 列表
Records = list[tuple[str, list[str]]]


def save(path: str | Path, records: Records) -> None:
    """原子写盘：先写同目录临时文件，再 `os.replace` 顶上。

    写一半崩溃时顶不上去，目标文件保持旧版完整 —— 调用方最坏情况是
    拿到上一版索引，而不是一个坏文件。
    """
    target = Path(path)
    payload = {"version": _FORMAT_VERSION, "records": records}
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, suffix=".pkl.tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(payload, f)
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def load(path: str | Path) -> Records:
    """读回 `save` 写的数据。文件损坏 / 格式版本不对抛 `ValueError`。"""
    with open(path, "rb") as f:
        try:
            payload = pickle.load(f)
        except Exception as exc:  # 坏文件什么错都可能抛，统一翻译成 ValueError
            raise ValueError(f"pickle 文件读不出来（{path}）：{exc}") from exc

    if (
        not isinstance(payload, dict)
        or payload.get("version") != _FORMAT_VERSION
        or not isinstance(payload.get("records"), list)
    ):
        raise ValueError(
            f"pickle 文件不是本模块写的或版本不认（{path}）："
            f"期望 version={_FORMAT_VERSION} 的数据文件"
        )
    return payload["records"]
