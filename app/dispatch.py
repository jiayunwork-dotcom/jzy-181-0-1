"""多时段经济调度业务层：调用 QP 内核、处理网损罚因子迭代、不可行性检测、
组装对用户友好的结果（λ、各约束状态与乘子、最优性校验）。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .qp import solve_qp, solve_qp_elastic, QPError, BOUND_TOL

LOSS_ITER_MAX = 40
LOSS_TOL = 1e-8


class InfeasibleError(RuntimeError):
    def __init__(self, periods, shortfalls):
        self.periods = periods
        self.shortfalls = shortfalls
        super().__init__("infeasible")


@dataclass
class DispatchResult:
    P: np.ndarray                  # (N,T)
    lam: np.ndarray               # (T,)
    cost: float
    status: str
    flags: dict                    # 各时段各机碰壁/爬坡标记
    multipliers: dict              # 爬坡乘子
    iterations: int
    # 网损
    loss: np.ndarray | None = None
    penalty_iters: int = 0
    imbalance: np.ndarray | None = None
    # 不可行（弹性）信息
    feasible: bool = True
    shortfall: np.ndarray | None = None


def units_for_kernel(unit_groups):
    return [
        {"name": g["name"], "pmin": g["pmin"], "pmax": g["pmax"],
         "a": g["a"], "b": g["b"], "c": g["c"],
         "init": g["init"], "ru": g["ru"], "rd": g["rd"]}
        for g in unit_groups
    ]


def total_cost(P, units):
    a = np.array([u["a"] for u in units], float)
    b = np.array([u["b"] for u in units], float)
    c = np.array([u["c"] for u in units], float)
    return float(np.sum(a[:, None] + b[:, None] * P + c[:, None] * P ** 2))


def loss_B(P, B0, B1, B2):
    return np.sum(P * (B0 @ P), axis=0) + B1 @ P + B2


def _feasibility_check(units, T, load):
    """弹性规划求解；可行返回 None，不可行返回 (periods, shortfalls, P_top)。"""
    ku = units_for_kernel(units)
    res = solve_qp_elastic(ku, T, load, tol=1e-8, max_iter=200)
    if getattr(res, "feasible", True):
        return None
    v = np.maximum(res.v, 0.0)
    periods = [int(t) for t in np.where(v > 1e-4)[0]]
    return periods, v, res.P


def economic_dispatch(units, T, load, B=None):
    """主入口。units 为校验后的机组群（保留校验时刻参数版本），load (T,)，
    B 为可选 dict(B0,B1,B2)。返回 DispatchResult。"""
    load = np.asarray(load, dtype=float).reshape(T)
    ku = units_for_kernel(units)

    # 1) 先做可行性检测
    feas = _feasibility_check(units, T, load)
    if feas is not None:
        periods, shortfall, Ptop = feas
        raise _make_infeasible(periods, shortfall, Ptop, units, T, load)

    # 2) 正式求解（含网损迭代）
    if B is None:
        res = solve_qp(ku, T, load, tol=1e-8, max_iter=300)
        return _package(res, units, T, load, None, 0)

    return _dispatch_with_losses(ku, units, T, load, B)


def _dispatch_with_losses(ku, units, T, load, B):
    B0, B1, B2 = B["B0"], B["B1"], B["B2"]
    P = None
    J = np.ones((len(ku), T))
    off = np.zeros(T)
    last_imbalance = None
    for it in range(1, LOSS_ITER_MAX + 1):
        if P is not None:
            grad = (B0 + B0.T) @ P + B1[:, None]
            J = 1.0 - grad
            Pl = loss_B(P, B0, B1, B2)
            off = Pl - np.sum(grad * P, axis=0)  # Ploss - Σ ∂Ploss/∂P · P
        res = solve_qp(ku, T, load, J=J, off=off, tol=1e-9, max_iter=200)
        P = res.P
        Pl = loss_B(P, B0, B1, B2)
        imb = np.sum(P, axis=0) - Pl - load
        last_imbalance = imb
        if np.max(np.abs(imb)) <= LOSS_TOL:
            pkg = _package(res, units, T, load, B, it)
            pkg.loss = Pl
            pkg.imbalance = imb
            return pkg
    # 迭代到上限仍不收敛：不返回半截解
    raise QPError(f"网损罚因子迭代 {LOSS_ITER_MAX} 次后功率不平衡仍为 "
                  f"{np.max(np.abs(last_imbalance)):.3e} MW")


def _package(res, units, T, load, B, loss_iters):
    N = len(units)
    P = res.P
    c = np.array([u["c"] for u in units], float)
    b = np.array([u["b"] for u in units], float)

    flags = {
        "at_high": res.at_limit_high().tolist(),
        "at_low": res.at_limit_low().tolist(),
        "ramp_up_active": res.ramp_up_active().tolist(),
        "ramp_down_active": res.ramp_down_active().tolist(),
    }
    multipliers = {
        "ramp_up": np.maximum(res.z_ru, 0.0).tolist(),
        "ramp_down": np.maximum(res.z_rd, 0.0).tolist(),
        "upper": np.maximum(res.z_hi, 0.0).tolist(),
        "lower": np.maximum(res.z_lo, 0.0).tolist(),
        "lambda": res.lam.tolist(),
    }
    cost = total_cost(P, units)
    loss = None
    imbalance = None
    if B is not None:
        loss = loss_B(P, B["B0"], B["B1"], B["B2"])
        imbalance = np.sum(P, axis=0) - loss - load

    return DispatchResult(
        P=P, lam=res.lam, cost=cost, status="optimal",
        flags=flags, multipliers=multipliers, iterations=int(res.iterations),
        loss=loss, penalty_iters=loss_iters, imbalance=imbalance,
        feasible=True, shortfall=None)


def _make_infeasible(periods, shortfall, Ptop, units, T, load):
    ex = InfeasibleError(periods, {int(t): float(shortfall[t]) for t in periods})
    ex.attainable_generation = {int(t): float(np.sum(Ptop[:, t])) for t in periods}
    ex.load = {int(t): float(load[t]) for t in periods}
    ex.P_top = Ptop
    return ex


# ---------------- KKT / 最优性校验 ----------------

def verify_kkt(dr: DispatchResult, units, load, B=None, tol=1e-6):
    """逐条检验最优性条件，返回 dict(ok, violations)。"""
    P = dr.P
    N, T = P.shape
    b = np.array([u["b"] for u in units], float)
    c = np.array([u["c"] for u in units], float)
    lam = dr.lam
    zru = np.array(dr.multipliers["ramp_up"])
    zrd = np.array(dr.multipliers["ramp_down"])
    zhi = np.array(dr.multipliers["upper"])
    zlo = np.array(dr.multipliers["lower"])
    violations = []

    active_hi = np.array(dr.flags["at_high"])
    active_lo = np.array(dr.flags["at_low"])
    active_ru = np.array(dr.flags["ramp_up_active"])
    active_rd = np.array(dr.flags["ramp_down_active"])

    # 罚因子
    if B is None:
        J = np.ones_like(P)
    else:
        grad = (B["B0"] + B["B0"].T) @ P + B["B1"][:, None]
        J = 1.0 - grad

    # 平稳: b+2cP + (爬坡乘子净贡献) + 上下界乘子 = J·λ
    for i in range(N):
        for t in range(T):
            net = 0.0
            # 上爬坡行新时刻=t 系数 +1，旧列(t-1)系数 -1
            net += zru[i, t]
            net -= zru[i, t + 1] if t + 1 < T else 0.0
            net -= zrd[i, t]
            net += zrd[i, t + 1] if t + 1 < T else 0.0
            net += zhi[i, t] - zlo[i, t]
            mc = b[i] + 2 * c[i] * P[i, t]
            lhs = mc + net
            if abs(lhs - J[i, t] * lam[t]) > tol * (1 + abs(lam[t])):
                # 未碰壁机组才要求严格等于；碰壁机组由不等式侧处理
                if not (active_hi[i, t] or active_lo[i, t]):
                    violations.append(
                        f"unit{i} t{t}: 修正微增率 {lhs:.6f} != λ {J[i,t]*lam[t]:.6f}")

    # 顶上限：修正微增率 <= λ
    for i in range(N):
        for t in range(T):
            if active_hi[i, t]:
                mc = b[i] + 2 * c[i] * P[i, t]
                net = (zru[i, t] - (zru[i, t+1] if t+1 < T else 0)
                       - zrd[i, t] + (zrd[i, t+1] if t+1 < T else 0)
                       - zlo[i, t])
                if mc + net > J[i, t] * lam[t] + tol * (1 + abs(lam[t])):
                    violations.append(f"unit{i} t{t} 顶上限但修正微增率>λ")
            if active_lo[i, t]:
                mc = b[i] + 2 * c[i] * P[i, t]
                net = (zru[i, t] - (zru[i, t+1] if t+1 < T else 0)
                       - zrd[i, t] + (zrd[i, t+1] if t+1 < T else 0)
                       + zhi[i, t])
                if mc + net < J[i, t] * lam[t] - tol * (1 + abs(lam[t])):
                    violations.append(f"unit{i} t{t} 贴下限但修正微增率<λ")

    # 乘子非负
    for nm, arr in (("ramp_up", zru), ("ramp_down", zrd), ("upper", zhi), ("lower", zlo)):
        if np.any(arr < -tol):
            violations.append(f"乘子 {nm} 出现负值")

    # 互补松弛（活动约束允许正乘子，非活动应近零）
    for nm, active, mult in (("ramp_up", active_ru, zru), ("ramp_down", active_rd, zrd),
                             ("upper", active_hi, zhi), ("lower", active_lo, zlo)):
        inactive = ~active
        if np.any(np.abs(mult[inactive]) > max(tol, 1e-4)):
            violations.append(f"非活动约束 {nm} 乘子非零")

    # 功率平衡
    if B is None:
        imb = np.sum(P, axis=0) - load
    else:
        Pl = loss_B(P, B["B0"], B["B1"], B["B2"])
        imb = np.sum(P, axis=0) - Pl - load
    if np.max(np.abs(imb)) > 1e-6:
        violations.append(f"功率不平衡 {np.max(np.abs(imb)):.2e}")

    return {"ok": len(violations) == 0, "violations": violations}
