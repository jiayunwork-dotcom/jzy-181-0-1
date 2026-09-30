"""参数校验：所有错误返回带字段名的 ValidationError。"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


class ValidationError(ValueError):
    """带出错字段名的校验错误。"""

    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")


def _is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and math.isfinite(float(x))


def _is_finite_array(a: Any) -> bool:
    try:
        arr = np.asarray(a, dtype=float)
    except (TypeError, ValueError):
        return False
    return arr.size > 0 and np.all(np.isfinite(arr))


def validate_unit(unit: dict) -> dict:
    """校验并归一化单台机组参数，返回干净的 dict。"""
    if not isinstance(unit, dict):
        raise ValidationError("unit", "必须是对象")

    name = unit.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValidationError("name", "机组名必须是非空字符串")

    cleaned: dict[str, Any] = {"name": name}

    for field in ("Pmin", "pmax", "a", "b", "c", "initial_output", "init"):
        if field in unit and field not in ("Pmin", "pmax", "initial_output", "init"):
            pass

    # 兼容 Pmin/pmin 两种写法（外部 API 用 Pmin，内核用 pmin）
    pmin_v = unit.get("Pmin", unit.get("pmin"))
    pmax_v = unit.get("pmax", unit.get("Pmax"))
    init_v = unit.get("initial_output", unit.get("init"))
    if pmin_v is None:
        raise ValidationError("Pmin", "缺少 Pmin")
    if pmax_v is None:
        raise ValidationError("pmax", "缺少 pmax")
    if init_v is None:
        raise ValidationError("initial_output", "缺少 initial_output")

    for field, val in (("Pmin", pmin_v), ("pmax", pmax_v),
                       ("initial_output", init_v)):
        if not _is_finite_number(val):
            raise ValidationError(field, "必须是有限数")
        if float(val) < 0:
            raise ValidationError(field, "不能为负")

    pmin_f, pmax_f, init_f = float(pmin_v), float(pmax_v), float(init_v)
    if pmin_f > pmax_f:
        raise ValidationError("Pmin", f"Pmin({pmin_f}) 不能大于 pmax({pmax_f})")
    if init_f < pmin_f - 1e-9 or init_f > pmax_f + 1e-9:
        raise ValidationError(
            "initial_output",
            f"初始出力({init_f}) 超出 [{pmin_f}, {pmax_f}]")

    for field in ("a", "b", "c"):
        val = unit.get(field)
        if val is None:
            raise ValidationError(field, f"缺少成本系数 {field}")
        if not _is_finite_number(val):
            raise ValidationError(field, "必须是有限数")
        if float(val) < 0:
            raise ValidationError(field, "成本系数不能为负")
    c_f = float(unit["c"])
    if c_f == 0.0 and float(unit["b"]) == 0.0 and float(unit["a"]) == 0.0:
        raise ValidationError("c", "c=0 的线性机组 b 不能同时为 0（需要有效边际成本）")

    ru_raw = unit.get("ramp_up", unit.get("ru"))
    rd_raw = unit.get("ramp_down", unit.get("rd"))
    if ru_raw is None:
        raise ValidationError("ramp_up", "缺少 ramp_up")
    if rd_raw is None:
        raise ValidationError("ramp_down", "缺少 ramp_down")

    def _clean_rate(field, val):
        if isinstance(val, (list, tuple, np.ndarray)):
            arr = np.asarray(val, dtype=float)
            if arr.ndim != 1 or np.any(~np.isfinite(arr)) or np.any(arr <= 0):
                raise ValidationError(field, "爬坡数组必须是正数一维有限数组")
            return arr.tolist()
        if not _is_finite_number(val) or float(val) <= 0:
            raise ValidationError(field, "爬坡速率必须是正数")
        return float(val)

    cleaned.update({
        "pmin": pmin_f, "pmax": pmax_f, "init": init_f,
        "a": float(unit["a"]), "b": float(unit["b"]), "c": c_f,
        "ru": _clean_rate("ramp_up", ru_raw),
        "rd": _clean_rate("ramp_down", rd_raw),
    })
    return cleaned


def validate_b_coeff(B: Any, n_units: int) -> np.ndarray | None:
    """校验网损 B 系数（可选）。返回 (n,n) ndarray 或 None。"""
    if B is None:
        return None
    try:
        arr = np.asarray(B, dtype=float)
    except (TypeError, ValueError):
        raise ValidationError("B", "B 系数必须是数值矩阵")
    if arr.shape != (n_units, n_units):
        raise ValidationError("B", f"B 必须是 {n_units}×{n_units} 方阵，实际 {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValidationError("B", "B 含非有限元素")
    if not np.allclose(arr, arr.T, atol=1e-10):
        raise ValidationError("B", "B 必须是对称矩阵")
    return arr


def validate_load(loads: Any, n_periods: int) -> np.ndarray:
    try:
        arr = np.asarray(loads, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raise ValidationError("load", "负荷曲线必须是数值数组")
    if arr.size != n_periods:
        raise ValidationError("load", f"负荷时段数({arr.size})与声明({n_periods})不符")
    if not np.all(np.isfinite(arr)):
        raise ValidationError("load", "负荷含非有限数")
    return arr
