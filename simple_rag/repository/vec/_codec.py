"""向量 → sqlite vector 字节串 —— 归一化 + dtype 统一（纯函数）。

实测（探针 09）这两条不做会**静默**出事：

- dtype 写错：`float64` 的字节被 vec0 照单全收，存进去是垃圾向量，不报错
- NaN / Inf：被接受，且污染 KNN 排序（NaN 行排最前）

零向量那一条是**数学推导**（`0/0` → NaN），不是实测。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def vector_to_sqlite_vector(
    vector: Sequence[float] | np.ndarray, dim: int
) -> bytes:
    """把向量转成 sqlite vector（vec0 的 `float[dim]` 列）接受的 float32 字节串。

    **已 L2 归一化**，归一化无条件执行，没有开关。查询向量也走这里 ——
    两端共用同一条路径，免得出现「一端归一化另一端没有」这类隐性假设。

    Args:
        vector: 一维数值序列。接受 `list[float]`、`tuple[float]` 和
            `numpy.ndarray`（`int32` / `float64` 等 dtype 都会被转成 float32）。
            **不接受已经 `tobytes()` 过的 `bytes`** —— numpy 会把它当字符串解析
            并抛 `ValueError`，所以「先编码再传进来」是错的。
        dim: 建表时声明的向量维度，必须与 `vector` 的长度精确一致。

    Returns:
        长度为 `dim * 4` 的 float32 小端字节串，模长已归一化为 1。

    Raises:
        ValueError: 维度不是 1 维（标量、嵌套列表都算）、长度与 `dim` 不符、
            含 NaN / Inf，或者是零向量。
        TypeError: `vector` 转不成数值数组（如 `dict`、生成器）。
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
