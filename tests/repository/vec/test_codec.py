"""`_codec` 的纯单测 —— 不碰数据库。"""

import numpy as np
import pytest

from simple_rag.repository.vec._codec import vector_to_sqlite_vector

DIM = 4


def decode(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def test_结果是_float32_且已归一化():
    blob = vector_to_sqlite_vector([3.0, 4.0, 0.0, 0.0], DIM)
    arr = decode(blob)
    assert arr.dtype == np.float32
    assert arr.shape == (DIM,)
    assert np.isclose(np.linalg.norm(arr), 1.0)
    assert np.allclose(arr, [0.6, 0.8, 0.0, 0.0])


def test_float64_输入被转换而不是照单全收():
    """实测（探针 09）dtype 写错会静默损坏数据，所以这里必须转。"""
    blob = vector_to_sqlite_vector(
        np.array([3.0, 4.0, 0.0, 0.0], dtype=np.float64), DIM
    )
    assert len(blob) == DIM * 4  # float32，不是 float64 的 8 字节
    assert np.allclose(decode(blob), [0.6, 0.8, 0.0, 0.0])


def test_int32_输入被转换():
    blob = vector_to_sqlite_vector(np.array([0, 3, 4, 0], dtype=np.int32), DIM)
    assert len(blob) == DIM * 4
    assert np.allclose(decode(blob), [0.0, 0.6, 0.8, 0.0])


def test_归一化对方向不变_只改长度():
    short = decode(vector_to_sqlite_vector([1.0, 0.0, 0.0, 0.0], DIM))
    long = decode(vector_to_sqlite_vector([100.0, 0.0, 0.0, 0.0], DIM))
    assert np.allclose(short, long)


def test_零向量被拦():
    """0/0 会变成 NaN，所以必须在除法**之前**拦（否则绕过 isfinite 检查）。"""
    with pytest.raises(ValueError, match="零向量"):
        vector_to_sqlite_vector([0.0] * DIM, DIM)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_NaN_和_Inf_被拦(bad):
    with pytest.raises(ValueError, match="NaN"):
        vector_to_sqlite_vector([1.0, bad, 0.0, 0.0], DIM)


def test_维度不符报错():
    with pytest.raises(ValueError, match="维度不符"):
        vector_to_sqlite_vector([1.0, 0.0, 0.0], DIM)


def test_二维输入报错():
    with pytest.raises(ValueError, match="一维"):
        vector_to_sqlite_vector([[1.0, 0.0, 0.0, 0.0]], DIM)


def test_标量输入报错():
    with pytest.raises(ValueError, match="一维"):
        vector_to_sqlite_vector(1.0, DIM)


def test_转不动的输入由_numpy_抛_模块不包装():
    # spec §6：这两类落到异常契约的同一格，不额外包一层
    with pytest.raises(TypeError):
        vector_to_sqlite_vector({"a": 1}, DIM)
    with pytest.raises(ValueError):
        vector_to_sqlite_vector(["a", "b", "c", "d"], DIM)
