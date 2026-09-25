"""向量编码 —— 归一化 + dtype 统一（纯函数）。

实测（探针 09）这两条不做会**静默**出事：

- dtype 写错：`float64` 的字节被 vec0 照单全收，存进去是垃圾向量，不报错
- NaN / Inf：被接受，且污染 KNN 排序（NaN 行排最前）

零向量那一条是**数学推导**（`0/0` → NaN），不是实测。
"""

from __future__ import annotations

import numpy as np


def encode(vector, dim: int) -> bytes:
    """把向量转成 vec0 接受的 float32 字节串，**已 L2 归一化**。

    归一化无条件执行，没有开关。查询向量也走这里 —— 两端共用同一条路径，
    免得出现「一端归一化另一端没有」这类隐性假设。
    """
    # 统一转换，不依赖库的校验。转不动的输入由 numpy 自己抛
    # （{"a": 1} -> TypeError，["a", "b"] -> ValueError），本模块不包装
    arr = np.asarray(vector, dtype=np.float32)

    if arr.ndim != 1:
        raise ValueError(f"向量必须是一维的，收到 {arr.ndim} 维")
    if arr.shape[0] != dim:
        raise ValueError(f"向量维度不符：期望 {dim}，收到 {arr.shape[0]}")

    norm = float(np.linalg.norm(arr))
    # 零向量必须在除法**之前**拦：0/0 会变成 NaN，反而绕过 isfinite 检查
    if not np.isfinite(arr).all() or norm == 0.0:
        raise ValueError("向量含 NaN / Inf，或是零向量")

    return (arr / norm).astype(np.float32).tobytes()
