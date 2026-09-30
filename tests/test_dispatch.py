"""核心求解器与业务逻辑回归测试。"""
import numpy as np
import pytest

from app.qp import solve_qp, solve_qp_elastic
from app.dispatch import economic_dispatch, verify_kkt
from app.validation import validate_unit, ValidationError
from app.relocal import recompute
from .fixtures import *


# ---------------- 手算结果 ----------------

def test_single_period_handcalc_150():
    r = solve_qp(SINGLE_UNITS, 1, np.array([SINGLE_LOAD_150]), tol=1e-11)
    assert np.allclose(r.P.ravel(), SINGLE_EXPECTED_150, atol=1e-6)
    assert abs(r.lam[0] - SINGLE_LAMBDA_150) < 1e-6


def test_single_period_handcalc_180_high_bound():
    r = solve_qp(SINGLE_UNITS, 1, np.array([SINGLE_LOAD_180]), tol=1e-11)
    assert np.allclose(r.P.ravel(), SINGLE_EXPECTED_180, atol=1e-5)
    assert abs(r.lam[0] - SINGLE_LAMBDA_180) < 1e-4
    assert r.at_limit_high()[2, 0]
    assert r.z_hi[2, 0] > 0


def test_multi_period_handcalc():
    r = solve_qp(MULTI_UNITS, 4, MULTI_LOADS, tol=1e-9)
    assert np.allclose(r.P, MULTI_EXPECTED, atol=1e-5)
    assert np.allclose(r.lam, MULTI_LAMBDA, atol=1e-5)
    # G1 在 t0 被初始爬坡限制顶在 40
    from .fixtures import MULTI_RAMP_BIND, MULTI_RAMP_MULT
    i, t = MULTI_RAMP_BIND
    assert r.ramp_up_active()[i, t]
    assert abs(r.z_ru[i, t] - MULTI_RAMP_MULT) < 1e-4


# ---------------- 平衡精度 1e-6 ----------------

def test_power_balance_within_1e6():
    r = solve_qp(MULTI_UNITS, 4, MULTI_LOADS, tol=1e-9)
    imb = np.abs(r.P.sum(axis=0) - MULTI_LOADS)
    assert np.max(imb) < 1e-6


def test_power_balance_single():
    for L in (50.0, 150.0, 180.0, 220.0):
        r = solve_qp(SINGLE_UNITS, 1, np.array([L]), tol=1e-11)
        assert abs(r.P.sum() - L) < 1e-6


# ---------------- 爬坡足够大退化为逐时段独立等微增 ----------------

def test_large_ramp_decouples():
    big = []
    for u in SINGLE_UNITS:
        uu = dict(u)
        uu["ru"], uu["rd"], uu["init"] = 1.0e5, 1.0e5, 20.0
        big.append(uu)
    loads = np.array([100.0, 150.0, 180.0, 60.0])
    r = solve_qp(big, 4, loads, tol=1e-11)
    for t, L in enumerate(loads):
        single = solve_qp(SINGLE_UNITS, 1, np.array([L]), tol=1e-11)
        assert np.allclose(r.P[:, t], single.P.ravel(), atol=1e-6)
        assert abs(r.lam[t] - single.lam[0]) < 1e-6


# ---------------- 单时段增量按 1/c 分配 ----------------

def test_marginal_increment_proportional_to_inverse_c():
    # 内部机组（不碰壁）从负荷 L 到 L+ΔL，增量与 1/c 成正比
    La = 150.0
    dL = 1.0  # 小增量，保证三机都在内部（G3 距上限仅 1.25MW）
    rA = solve_qp(SINGLE_UNITS, 1, np.array([La]), tol=1e-10)
    rB = solve_qp(SINGLE_UNITS, 1, np.array([La + dL]), tol=1e-10)
    dP = (rB.P - rA.P).ravel()
    cinv = np.array([1 / u["c"] for u in SINGLE_UNITS])
    expected = dL * cinv / cinv.sum()   # 增量按 1/c 比例分配，合计等于 dL
    assert np.allclose(dP, expected, atol=1e-6)
    assert abs(dP.sum() - dL) < 1e-7


# ---------------- b、c 同比缩放，出力不变、λ 同比 ----------------

def test_scaling_bc_preserves_dispatch():
    k = 2.7
    scaled = []
    for u in SINGLE_UNITS:
        uu = dict(u)
        uu["b"], uu["c"] = k * u["b"], k * u["c"]
        scaled.append(uu)
    r1 = solve_qp(SINGLE_UNITS, 1, np.array([150.0]), tol=1e-10)
    r2 = solve_qp(scaled, 1, np.array([150.0]), tol=1e-10)
    assert np.allclose(r1.P, r2.P, atol=1e-7)
    assert np.allclose(r1.lam * k, r2.lam, rtol=1e-7)


# ---------------- 网损系数全零退化为不计网损 ----------------

def test_zero_loss_coeffs():
    n = 3
    B = {"B0": np.zeros((n, n)), "B1": np.zeros(n), "B2": 0.0}
    dr0 = economic_dispatch(SINGLE_UNITS, 1, np.array([150.0]))
    drB = economic_dispatch(SINGLE_UNITS, 1, np.array([150.0]), B=B)
    assert np.allclose(dr0.P, drB.P, atol=1e-7)
    assert np.allclose(dr0.lam, drB.lam, atol=1e-7)
    assert np.max(np.abs(drB.loss)) < 1e-9


def test_loss_iteration_converges_and_balances():
    units = [
        {"name": "G1", "pmin": 0.0, "pmax": 200.0, "a": 0.0, "b": 4.0, "c": 0.02,
         "ru": 1e5, "rd": 1e5, "init": 50.0},
        {"name": "G2", "pmin": 0.0, "pmax": 200.0, "a": 0.0, "b": 6.0, "c": 0.03,
         "ru": 1e5, "rd": 1e5, "init": 50.0},
    ]
    B = {"B0": np.array([[1e-4, 0.0], [0.0, 2e-4]]), "B1": np.zeros(2), "B2": 0.0}
    dr = economic_dispatch(units, 1, np.array([150.0]), B=B)
    assert dr.penalty_iters <= 40
    assert np.max(np.abs(dr.imbalance)) < 1e-6
    # 网损为正，总发电 > 负荷
    assert np.sum(dr.P) > 150.0
    kkt = verify_kkt(dr, units, np.array([150.0]), B=B)
    assert kkt["ok"], kkt["violations"]


# ---------------- 最优性条件 KKT ----------------

def test_kkt_conditions():
    dr = economic_dispatch(MULTI_UNITS, 4, MULTI_LOADS)
    kkt = verify_kkt(dr, MULTI_UNITS, MULTI_LOADS)
    assert kkt["ok"], kkt["violations"]


def test_kkt_linear_unit():
    units = [
        {"name": "A", "pmin": 0.0, "pmax": 100.0, "a": 0.0, "b": 4.0, "c": 0.0,
         "ru": 1e5, "rd": 1e5, "init": 50.0},
        {"name": "B", "pmin": 0.0, "pmax": 100.0, "a": 0.0, "b": 6.0, "c": 0.0,
         "ru": 1e5, "rd": 1e5, "init": 50.0},
    ]
    dr = economic_dispatch(units, 1, np.array([150.0]))
    assert np.allclose(dr.P.ravel(), [100.0, 50.0], atol=1e-6)
    assert abs(dr.lam[0] - 6.0) < 1e-6
    assert dr.flags["at_high"][0][0]
    kkt = verify_kkt(dr, units, np.array([150.0]))
    assert kkt["ok"], kkt["violations"]


# ---------------- 局部重算与全量一致 ----------------

def test_local_recompute_matches_full_random():
    rng = np.random.default_rng(42)
    units = [
        {"name": "G1", "pmin": 0.0, "pmax": 120.0, "a": 0.0, "b": 4.0, "c": 0.4,
         "ru": 60.0, "rd": 60.0, "init": 40.0},
        {"name": "G2", "pmin": 0.0, "pmax": 120.0, "a": 0.0, "b": 6.0, "c": 0.25,
         "ru": 80.0, "rd": 80.0, "init": 40.0},
        {"name": "G3", "pmin": 0.0, "pmax": 120.0, "a": 0.0, "b": 8.0, "c": 0.15,
         "ru": 90.0, "rd": 90.0, "init": 40.0},
    ]
    T = 12
    base = np.array([120.0, 160.0, 200.0, 150.0, 100.0, 180.0,
                     220.0, 170.0, 130.0, 90.0, 140.0, 190.0])
    dr0 = economic_dispatch(units, T, base)
    for trial in range(8):
        new = base.copy()
        idxs = rng.choice(T, size=rng.integers(1, 4), replace=False)
        new[idxs] += rng.uniform(-30, 30, size=idxs.size)
        new = np.clip(new, 20, 300)
        pdr, window = recompute(dr0, units, base, new, list(idxs))
        full = economic_dispatch(units, T, new)
        assert np.max(np.abs(pdr.P - full.P)) < 1e-4
        assert abs(pdr.cost - full.cost) / max(abs(full.cost), 1e-8) < 1e-8


# ---------------- 平均分摊成本高于服务解 ----------------

def test_average_dispatch_more_expensive():
    units = MULTI_UNITS
    dr = economic_dispatch(units, 4, MULTI_LOADS)
    P = dr.P
    # 平均分摊：各机组按容量比例承担每时段负荷，且满足爬坡（用逐时段 clip 近似可行）
    cap = np.array([u["pmax"] for u in units])
    Pavg = np.zeros_like(P)
    prev = np.array([u["init"] for u in units])
    for t in range(4):
        share = MULTI_LOADS[t] * cap / cap.sum()
        lo = np.maximum(0.0, prev - np.array([u["rd"] for u in units]))
        hi = np.minimum(cap, prev + np.array([u["ru"] for u in units]))
        x = np.clip(share, lo, hi)
        # 若 clip 导致总量偏差，再按余量在可调节机组间补
        diff = MULTI_LOADS[t] - x.sum()
        room_hi = hi - x
        room_lo = x - lo
        if diff > 0 and room_hi.sum() > 0:
            x = x + diff * room_hi / room_hi.sum()
        elif diff < 0 and room_lo.sum() > 0:
            x = x + diff * room_lo / room_lo.sum()
        Pavg[:, t] = x
        prev = x
    # 平均方案须满足爬坡
    for i, u in enumerate(units):
        assert abs(Pavg[i, 0] - u["init"]) <= u["ru"] + 1e-6
        for t in range(1, 4):
            assert abs(Pavg[i, t] - Pavg[i, t - 1]) <= u["ru"] + 1e-6
    b = np.array([u["b"] for u in units])
    c = np.array([u["c"] for u in units])

    def cost(Px):
        return float(np.sum(b[:, None] * Px + c[:, None] * Px ** 2))

    assert cost(Pavg) > dr.cost + 1e-6


# ---------------- 不可行检测 ----------------

def test_infeasible_period_reported():
    from app.dispatch import InfeasibleError
    units = [
        {"name": "G1", "pmin": 0.0, "pmax": 100.0, "a": 0.0, "b": 4.0, "c": 0.5,
         "ru": 40.0, "rd": 40.0, "init": 0.0},
        {"name": "G2", "pmin": 0.0, "pmax": 100.0, "a": 0.0, "b": 6.0, "c": 0.25,
         "ru": 1e4, "rd": 1e4, "init": 0.0},
        {"name": "G3", "pmin": 0.0, "pmax": 100.0, "a": 0.0, "b": 8.0, "c": 0.1,
         "ru": 1e4, "rd": 1e4, "init": 0.0},
    ]
    loads = np.array([60.0, 1000.0, 80.0, 50.0])
    with pytest.raises(InfeasibleError) as ei:
        economic_dispatch(units, 4, loads)
    ex = ei.value
    assert 1 in ex.periods
    # t1 最大可达：G1≤80（爬坡），G2/G3≤100 -> 280，缺口 720
    assert abs(ex.shortfalls[1] - 720.0) < 1.0
    assert abs(ex.attainable_generation[1] - 280.0) < 1.0
    assert ex.load[1] == 1000.0


# ---------------- 参数校验 ----------------

@pytest.mark.parametrize("field,mod", [
    ("Pmin", {"Pmin": 120.0}),
    ("pmax", {"pmax": -1.0}),
    ("c", {"c": -0.1}),
    ("ramp_up", {"ramp_up": 0.0}),
    ("initial_output", {"initial_output": 999.0}),
])
def test_invalid_params(field, mod):
    base = {"name": "G", "Pmin": 0.0, "pmax": 100.0, "a": 0.0, "b": 4.0, "c": 0.5,
            "ramp_up": 10.0, "ramp_down": 10.0, "initial_output": 50.0}
    base.update(mod)
    with pytest.raises(ValidationError) as ei:
        validate_unit(base)
    assert ei.value.field == field


def test_load_not_finite():
    from app.validation import validate_load
    with pytest.raises(ValidationError):
        validate_load([1.0, float("nan"), 3.0], 3)


def test_b_matrix_dimension_mismatch():
    from app.validation import validate_b_coeff
    with pytest.raises(ValidationError):
        validate_b_coeff([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], 2)


def test_b_matrix_asymmetric():
    from app.validation import validate_b_coeff
    with pytest.raises(ValidationError):
        validate_b_coeff([[1.0, 0.2], [0.0, 1.0]], 2)
