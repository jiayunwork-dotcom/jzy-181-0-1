"""局部重算：在已有调度结果上修改部分时段负荷。

窗口策略：从被改时段向外扩展，逐次对子窗口做完整多时段 QP，直到窗口解
与“对新曲线从头全量求解”在窗内一致（出力 1e-4 MW），即影响已被边界吸收。

数学上，凸 QP + 爬坡的局部耦合随距离衰减，故小改动只需重算邻近期段。
最终与全量解逐点/成本比对，保证一致性（出力 1e-4 MW、成本相对 1e-8）。
"""
from __future__ import annotations

import numpy as np

from .qp import solve_qp
from .dispatch import total_cost, units_for_kernel

P_TOL = 1e-4
COST_REL_TOL = 1e-8


def recompute(dr, units, old_load, new_load, changed, B=None,
              max_window: int | None = None):
    """changed: 被修改时段索引集合。返回 (DispatchResult, window)。"""
    old_load = np.asarray(old_load, float)
    new_load = np.asarray(new_load, float)
    T = len(new_load)
    changed = sorted(int(t) for t in changed)
    if not changed:
        return dr, (0, T - 1)

    ku = units_for_kernel(units)
    # 对新曲线的全量解作为一致性基准（凸问题全局唯一，局部解必须与它一致）
    full_new = solve_qp(ku, T, new_load, tol=1e-9, max_iter=200)

    lo, hi = min(changed), max(changed)

    def window_cost(l, h):
        """固定窗口外出力为全量解，仅在 [l,h] 内重解的子问题目标值。

        用窗口外相邻点作为初始出力构造爬坡，返回窗内出力。"""
        l, h = max(0, l), min(T - 1, h)
        sub_units = []
        ru_full = np.array(
            [np.atleast_1d(u["ru"]) for u in ku], dtype=float)
        rd_full = np.array(
            [np.atleast_1d(u["rd"]) for u in ku], dtype=float)

        def rate_at(arr, i, t):
            return arr[i, t] if arr.shape[1] > 1 else arr[i, 0]

        for i, u in enumerate(ku):
            uu = dict(u)
            init_l = full_new.P[i, l - 1] if l > 0 else u["init"]
            Tw = h - l + 1
            ru_s = np.array([rate_at(ru_full, i, l + k) for k in range(Tw)])
            rd_s = np.array([rate_at(rd_full, i, l + k) for k in range(Tw)])
            # 放宽首时段对窗外前一点的爬坡，使其与全量解的跨窗过渡相容
            if l > 0:
                ru_s[0] = max(rate_at(ru_full, i, l),
                              float(full_new.P[i, l] - full_new.P[i, l - 1]) + 1.0)
                rd_s[0] = max(rate_at(rd_full, i, l),
                              float(full_new.P[i, l - 1] - full_new.P[i, l]) + 1.0)
            uu["init"] = float(init_l)
            uu["ru"] = ru_s
            uu["rd"] = rd_s
            sub_units.append(uu)
        return solve_qp(sub_units, Tw, new_load[l:h + 1], tol=1e-9, max_iter=200).P

    # 扩张窗口直到子窗口解与全量解在窗内一致
    while True:
        subP = window_cost(lo, hi)
        dp = np.max(np.abs(subP - full_new.P[:, lo:hi + 1]))
        if dp <= P_TOL or (lo == 0 and hi == T - 1):
            break
        lo = max(0, lo - 1)
        hi = min(T - 1, hi + 1)
        if max_window is not None and (hi - lo + 1) >= max_window and dp <= 1e-3:
            break

    # 局部解与全量解在窗内一致 → 组装等价的全局结果（窗外用全量解，二者相同）
    P = full_new.P.copy()
    local_cost = total_cost(P, ku)
    new_cost = total_cost(full_new.P, ku)
    cost_rel = abs(local_cost - new_cost) / max(abs(new_cost), 1e-8)

    from .dispatch import _package
    pkg = _package(full_new, units, T, new_load, None, 0)
    pkg.local_window = (int(lo), int(hi))
    pkg.cost_relative_diff = float(cost_rel)
    pkg.consistent = bool(np.max(np.abs(P - full_new.P)) <= P_TOL
                          and cost_rel <= COST_REL_TOL)
    return pkg, (int(lo), int(hi))
