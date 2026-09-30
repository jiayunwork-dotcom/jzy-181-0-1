"""
多时段经济调度 QP 内核 —— 自编原始-对偶内点法（Mehrotra 预测-校正）。

求解问题（N 台机、T 个时段）：

    min  Σ_it (b_i P_it + c_i P_it²)
    s.t. Σ_i J_it P_it = L_t + o_t                  t=1..T   乘子 y_t (= λ_t)
         pmin_i ≤ P_it ≤ pmax_i
         爬坡:  P_i0 - init_i ≤ ru_i0,  init_i - P_i0 ≤ rd_i0
                P_it - P_i,t-1 ≤ ru_it,  P_i,t-1 - P_it ≤ rd_it   (t≥1)

线性化网损通过 J_it = 1 - ∂Ploss/∂P_it 与常数项 o_t 进入平衡约束。

不可行性检测（solve_qp_elastic）不对平衡做松弛内点，而是对平衡乘子 λ 做
单调二分：固定 λ 解“无平衡等式”子问题（enforce_balance=False，有效线性
成本 b-λ·J），发电随 λ 单调增；在远高于真实边际的 λ 上限仍发不足即判
不可行，缺口 v=负荷-可达最大发电。

结构利用：爬坡约束只在同一台机的相邻时段之间耦合，每台机的 KKT 系数阵
是三对角阵（批量 Thomas 算法，O(NT)）；消去机组块后每个时段只剩一个
关于 λ_t 的标量方程。不调用任何第三方优化器。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BOUND_TOL = 1e-7  # 判碰壁的松弛阈值 (MW)


class QPError(RuntimeError):
    pass


@dataclass
class QPResult:
    P: np.ndarray                 # (N, T)
    lam: np.ndarray               # (T,) 平衡约束乘子
    z_ru: np.ndarray              # (N, T) 上爬坡乘子
    z_rd: np.ndarray              # (N, T) 下爬坡乘子
    z_hi: np.ndarray              # (N, T) 上限乘子
    z_lo: np.ndarray              # (N, T) 下限乘子
    v: np.ndarray | None          # (T,) 弹性变量
    iterations: int
    gap: float
    rpri: float
    rdual: float
    warm: dict = field(default_factory=dict)

    _pmin: np.ndarray = None
    _pmax: np.ndarray = None
    _s_ru: np.ndarray = None
    _s_rd: np.ndarray = None

    def at_limit_high(self, tol: float = BOUND_TOL) -> np.ndarray:
        return self.P >= self._pmax[:, None] - tol

    def at_limit_low(self, tol: float = BOUND_TOL) -> np.ndarray:
        return self.P <= self._pmin[:, None] + tol

    def ramp_up_active(self, tol: float = BOUND_TOL) -> np.ndarray:
        return self._s_ru <= tol

    def ramp_down_active(self, tol: float = BOUND_TOL) -> np.ndarray:
        return self._s_rd <= tol


@dataclass
class _QPData:
    N: int
    T: int
    pmin: np.ndarray
    pmax: np.ndarray
    b: np.ndarray
    c: np.ndarray
    ru: np.ndarray       # (N, T)
    rd: np.ndarray
    init: np.ndarray     # (N,)
    load: np.ndarray     # (T,)
    J: np.ndarray        # (N, T)
    off: np.ndarray      # (T,)
    elastic: bool
    M: float
    vmax: np.ndarray     # (T,)
    enforce_balance: bool = True


def _thomas_solve_batch(l, d, u, rhs):
    """批量三对角求解。l,d,u:(N,T)（l[:,0] 不用），rhs:(N,T) 或 (N,T,K)。"""
    if rhs.ndim == 2:
        rhs = rhs[..., None]
        squeeze = True
    else:
        squeeze = False
    N, T, K = rhs.shape
    cp = np.empty_like(rhs)
    dp = np.empty_like(rhs)
    cp[:, 0, :] = (u[:, 0] / d[:, 0])[:, None]
    dp[:, 0, :] = rhs[:, 0, :] / d[:, 0, None]
    for t in range(1, T):
        den = d[:, t] - l[:, t] * cp[:, t - 1, 0]
        cp[:, t, :] = (u[:, t] / den)[:, None]
        dp[:, t, :] = (rhs[:, t, :] - l[:, t, None] * dp[:, t - 1, :]) / den[:, None]
    x = np.empty_like(rhs)
    x[:, -1, :] = dp[:, -1, :]
    for t in range(T - 2, -1, -1):
        x[:, t, :] = dp[:, t, :] - cp[:, t, :] * x[:, t + 1, :]
    return x[..., 0] if squeeze else x


def _inv_diag_tridiag(l, d, u):
    """各三对角阵逆的对角元：D_t = d_t - l_t²/D_{t-1}，返回 1/D。(N,T)"""
    N, T = d.shape
    ud = np.empty_like(d)
    ud[:, 0] = d[:, 0]
    for t in range(1, T):
        ud[:, t] = d[:, t] - l[:, t] ** 2 / ud[:, t - 1]
    return 1.0 / ud


def solve_qp(
    units: list[dict],
    T: int,
    load: np.ndarray,
    *,
    J: np.ndarray | None = None,
    off: np.ndarray | None = None,
    elastic: bool = False,
    M: float | None = None,
    warm: dict | None = None,
    tol: float = 1e-9,
    max_iter: int = 120,
    on_progress=None,
    enforce_balance: bool = True,
    lam_offset: np.ndarray | float | None = None,
) -> QPResult:
    N = len(units)
    col = lambda key: np.array([u[key] for u in units], dtype=float)
    pmin = col("pmin")
    pmax = col("pmax")
    b = col("b")
    c = col("c")
    init = col("init")

    def _rate(key):
        r = np.array([u[key] for u in units], dtype=float)
        if r.ndim == 1:
            r = np.repeat(r[:, None], T, axis=1)
        if r.shape != (N, T):
            raise QPError("ramp rate array shape mismatch")
        return r

    ru = _rate("ru")
    rd = _rate("rd")
    load = np.asarray(load, dtype=float).reshape(T)
    J = np.ones((N, T)) if J is None else np.asarray(J, dtype=float).reshape(N, T)
    off = np.zeros(T) if off is None else np.asarray(off, dtype=float).reshape(T)

    cost_scale = max(1.0, float(np.max(np.abs(b)) + 2.0 * np.max(np.abs(c)) * np.max(pmax)))
    if M is None:
        M = 1.0e3 * cost_scale
    # v 上界只需覆盖“负荷超出全部机组出力”的最大缺口；取宽松上界避免初始点贴界
    vmax = np.maximum(np.abs(load) + np.sum(pmax) + 100.0, 200.0)

    # 无平衡子问题（对偶二分）：对偶目标含 -λ·ΣJP，有效线性成本 b_eff=b-λ·J，
    # λ 越大机组越愿意增发。
    b_eff = b
    if (not enforce_balance) and lam_offset is not None:
        lam = np.broadcast_to(np.asarray(lam_offset, dtype=float), (T,)).copy()
        b_eff = b[:, None] - lam[None, :] * J
        b_eff = np.broadcast_to(b_eff, (N, T)).copy()

    data = _QPData(N, T, pmin, pmax, b_eff, c, ru, rd, init, load, J, off,
                   elastic, M, vmax, enforce_balance)

    # ---------- 初始化（负荷跟随 + 爬坡投影；Mehrotra 移位对偶起点） ----------
    # 先按容量比例分摊负荷，再做前向+后向爬坡/边界投影，得到既接近平衡
    # 又满足全部爬坡的原始点，显著改善大负荷摆动算例的收敛。
    span = np.maximum(pmax - pmin, 1e-8)

    def _project_forward(P0):
        P = np.empty((N, T))
        P[:, 0] = np.clip(P0[:, 0], np.maximum(pmin, init - rd[:, 0]),
                          np.minimum(pmax, init + ru[:, 0]))
        for t in range(1, T):
            P[:, t] = np.clip(P0[:, t], np.maximum(pmin, P[:, t - 1] - rd[:, t]),
                              np.minimum(pmax, P[:, t - 1] + ru[:, t]))
        return P

    target0 = (load + off if enforce_balance else
               np.broadcast_to(np.array(np.sum(0.5 * (pmin + pmax))), (T,)))
    mid = 0.5 * (pmin + pmax)
    P0 = np.empty((N, T))
    for t in range(T):
        # 容量比例分摊负荷，再向中点小幅收缩，避免恰好落在边界导致零松弛
        share = pmin + (target0[t] - pmin.sum()) * span / span.sum()
        share = np.clip(share, pmin, pmax)
        P0[:, t] = 0.85 * share + 0.15 * mid
        P0[:, t] = np.clip(P0[:, t], pmin + 1e-3, pmax - 1e-3)
    P = _project_forward(P0)
    # 后向投影修正爬坡的越界（前向可能在负荷下降沿越下爬坡）
    for _pass in range(2):
        for t in range(T - 2, -1, -1):
            P[:, t] = np.clip(P[:, t], np.maximum(pmin, P[:, t + 1] - rd[:, t + 1]),
                              np.minimum(pmax, P[:, t + 1] + ru[:, t + 1]))
        P = _project_forward(P)
    if warm is not None and warm.get("P") is not None:
        P = np.array(warm["P"], dtype=float).reshape(N, T)
        P = np.minimum(np.maximum(P, pmin[:, None] + 1e-10), pmax[:, None] - 1e-10)
    v = None
    if elastic:
        # 等式松弛 ΣJP+v=load+off、v≥0。初始 P 已经过爬坡可行投影；
        # v 直接取负荷-发电缺口（≥0），等式严格成立，不做任何破坏等式的偏移。
        v = np.clip(load + off - np.sum(J * P, axis=0), 0.0, vmax)
        if warm is not None and warm.get("v") is not None:
            v = np.minimum(np.maximum(np.array(warm["v"], dtype=float), 1e-10), vmax - 1e-10)

    true_slacks = _true_slacks(P, v, data)
    s_lo, s_hi, s_ru, s_rd, s_e, s_vlo, s_vhi = true_slacks

    def _positive(s):
        smin = np.min(s)
        if smin <= 1e-7:
            s = s + (-smin + 1.0)
        return s

    s_lo = _positive(s_lo); s_hi = _positive(s_hi)
    s_ru = _positive(s_ru); s_rd = _positive(s_rd)
    if elastic:
        s_vlo = _positive(s_vlo); s_vhi = _positive(s_vhi)

    z_lo = np.ones((N, T)); z_hi = np.ones((N, T))
    z_ru = np.ones((N, T)); z_rd = np.ones((N, T))
    z_e = None
    z_vlo = np.ones(T) if elastic else None
    z_vhi = np.ones(T) if elastic else None
    y = np.zeros(T)
    if warm is not None:
        y = np.array(warm.get("y", np.zeros(T)), dtype=float)
        for nm, arr in (("z_lo", z_lo), ("z_hi", z_hi), ("z_ru", z_ru), ("z_rd", z_rd)):
            if nm in warm:
                arr[...] = np.maximum(warm[nm], 1e-10)
        if elastic:
            for nm, arr in (("z_vlo", z_vlo), ("z_vhi", z_vhi)):
                if nm in warm:
                    arr[...] = np.maximum(warm[nm], 1e-10)

    if warm is None:
        # Mehrotra 风格初始对偶：在当前原始点用“近似等微增”估计 λ。
        bv0 = b if b.ndim == 2 else b[:, None]
        mc = bv0 + 2.0 * c[:, None] * P
        y = np.average(mc * J, axis=0, weights=np.maximum(J, 1e-12))

    # Mehrotra 中心互补起点：令各互补积 s∘z 均衡到同一 μ（取各 slack 的几何均值），
    # 乘子 z=μ/s。贴界行得到大乘子、宽松行得到小乘子，量级自动协调。
    slacks_all = [s_lo, s_hi, s_ru, s_rd]
    if elastic:
        slacks_all += [s_vlo, s_vhi]
    mu0 = float(np.exp(np.mean([np.log(max(float(np.min(s)), 1e-8))
                               for s in slacks_all])))

    def _init_z(z, s):
        z[...] = np.maximum(mu0 / np.maximum(s, 1e-8), 1e-8)

    _init_z(z_lo, s_lo); _init_z(z_hi, s_hi)
    _init_z(z_ru, s_ru); _init_z(z_rd, s_rd)
    if elastic:
        _init_z(z_vlo, s_vlo); _init_z(z_vhi, s_vhi)

    it = 0
    rp = rd_ = gap = np.inf
    for it in range(1, max_iter + 1):
        if on_progress is not None:
            on_progress(it)

        # ---- 仿射方向 ----
        aff = _direction(data, P, v, y,
                         s_lo, s_hi, s_ru, s_rd, s_e, s_vlo, s_vhi,
                         z_lo, z_hi, z_ru, z_rd, z_e, z_vlo, z_vhi,
                         corrector=None, enforce_balance=enforce_balance)
        alpha_p_aff, alpha_d_aff = _step_lengths(
            data, aff, s_lo, s_hi, s_ru, s_rd, s_e, s_vlo, s_vhi,
            z_lo, z_hi, z_ru, z_rd, z_e, z_vlo, z_vhi)
        mu = _mu(s_lo, s_hi, s_ru, s_rd, s_e, s_vlo, s_vhi,
                 z_lo, z_hi, z_ru, z_rd, z_e, z_vlo, z_vhi)
        mu_aff = _mu_after(alpha_p_aff, alpha_d_aff,
                           s_lo, s_hi, s_ru, s_rd, s_e, s_vlo, s_vhi,
                           z_lo, z_hi, z_ru, z_rd, z_e, z_vlo, z_vhi, aff)
        mu_aff = min(mu_aff, 1e100)
        sigma = min(1.0, (mu_aff / max(mu, 1e-300)) ** 3) if mu > 0 else 0.0

        # ---- 预测-校正方向 ----
        d = _direction(data, P, v, y,
                       s_lo, s_hi, s_ru, s_rd, s_e, s_vlo, s_vhi,
                       z_lo, z_hi, z_ru, z_rd, z_e, z_vlo, z_vhi,
                       corrector=(sigma * mu, aff),
                       enforce_balance=enforce_balance)
        alpha_p, alpha_d = _step_lengths(
            data, d, s_lo, s_hi, s_ru, s_rd, s_e, s_vlo, s_vhi,
            z_lo, z_hi, z_ru, z_rd, z_e, z_vlo, z_vhi)

        P += alpha_p * d["P"]
        y += alpha_d * d["y"]
        z_lo += alpha_d * d["z_lo"]
        z_hi += alpha_d * d["z_hi"]
        z_ru += alpha_d * d["z_ru"]
        z_rd += alpha_d * d["z_rd"]
        if elastic:
            v += alpha_p * d["v"]
            z_vlo += alpha_d * d["z_vlo"]
            z_vhi += alpha_d * d["z_vhi"]
        for arr in (z_lo, z_hi, z_ru, z_rd):
            np.maximum(arr, 1e-13, out=arr)
        if elastic:
            for arr in (z_vlo, z_vhi, v):
                np.maximum(arr, 1e-13, out=arr)

        # primal slack 直接由更新后的原始变量重算（严格正），使平移残差不累积。
        ts = _true_slacks(P, v, data)
        eps_s = 1e-11
        s_lo = np.maximum(ts[0], eps_s); s_hi = np.maximum(ts[1], eps_s)
        s_ru = np.maximum(ts[2], eps_s); s_rd = np.maximum(ts[3], eps_s)
        if elastic:
            s_vlo = np.maximum(ts[5], eps_s)
            s_vhi = np.maximum(ts[6], eps_s)
        for arr in (P, y, s_lo, s_hi, s_ru, s_rd, z_lo, z_hi, z_ru, z_rd):
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=1e12, neginf=0.0)
        if elastic:
            for arr in (v, s_vlo, s_vhi, z_vlo, z_vhi):
                np.nan_to_num(arr, copy=False, nan=0.0, posinf=1e12, neginf=0.0)
            v[:] = np.minimum(np.maximum(v, 0.0), vmax)

        rp, rd_, gap, r_bal = _residuals(
            data, P, v, y,
            s_lo, s_hi, s_ru, s_rd, None, s_vlo, s_vhi,
            z_lo, z_hi, z_ru, z_rd, None, z_vlo, z_vhi)
        bal_ok = True if not enforce_balance else \
            r_bal < tol * (1.0 + np.max(np.abs(load)))
        # gap 阈值与目标函数尺度（功率×边际）比较，避免被巨大的非活动爬坡 slack 干扰
        bv = b_eff if b_eff.ndim == 2 else b_eff[:, None]
        obj_scale = max(1.0,
                        float(np.sum(np.abs(load)) * cost_scale),
                        float(np.sum(np.abs(bv) * P) + np.sum(c[:, None] * P ** 2)))
        if rp < tol and bal_ok and rd_ < tol * cost_scale \
                and gap * _nrows(data) < tol * obj_scale:
            state = "optimal"
            break
    else:
        state = "max_iter"

    if state != "optimal":
        raise QPError(f"interior point failed to converge: {state}, "
                      f"rpri={rp:.3e}, rdual={rd_:.3e}, gap={gap:.3e}, iters={max_iter}")

    warm_pack = {"P": P.copy(), "y": y.copy(),
                 "z_lo": z_lo, "z_hi": z_hi, "z_ru": z_ru, "z_rd": z_rd}
    if elastic:
        warm_pack.update({"v": v, "z_vlo": z_vlo, "z_vhi": z_vhi})

    res = QPResult(P=P, lam=y, z_ru=z_ru, z_rd=z_rd, z_hi=z_hi, z_lo=z_lo,
                   v=v if elastic else None, iterations=it, gap=gap, rpri=rp, rdual=rd_,
                   warm=warm_pack)
    res._pmin, res._pmax = pmin, pmax
    res._s_ru, res._s_rd = s_ru, s_rd
    return res


def solve_qp_elastic(units, T, load, *, J=None, off=None,
                     M: float | None = None, tol: float = 1e-9,
                     max_iter: int = 200, on_progress=None, warm: dict | None = None):
    """弹性可行性求解：min 发电成本 + M·Σv，s.t. ΣJP+v=load+off、v≥0、边界+爬坡。

    用对 λ（平衡乘子）的二分/单调上升法。固定 λ 时 P 子问题是无平衡约束的
    多时段 QP（线性成本 b+λ·J），v 子问题为 v=max(0,load+off-ΣJP)。
    λ 定义域 [-M, ∞)：λ∈(-M,∞) 时 v=0；λ=-M 是不可行临界。
    实际用 λ≥0 搜索（发电边际为正），在 λ=M 处若仍 ΣJP<load+off 判不可行。
    返回 QPResult（v 为各时段缺口；lam=λ）。"""
    load = np.asarray(load, dtype=float).reshape(T)
    off = np.zeros(T) if off is None else np.asarray(off, dtype=float).reshape(T)
    J = np.ones((len(units), T)) if J is None else np.asarray(J, dtype=float).reshape(len(units), T)
    target = load + off
    pmax = np.array([u["pmax"] for u in units], dtype=float)
    b = np.array([u["b"] for u in units], dtype=float)
    c = np.array([u["c"] for u in units], dtype=float)
    mc_max = float(np.max(np.abs(b) + 2.0 * np.maximum(c, 0.0) * pmax))
    # λ 上限只需略大于“顶到出力上限所需的边际”（~b+2c·pmax）；取 20 倍，
    # 足以让所有机组顶物理/爬坡上限，又不触发大 λ 的数值刚性。
    Mpen = M if M is not None else max(1.0e2, 20.0 * mc_max)
    last_good_P = None

    def dispatch(lam):
        nonlocal last_good_P
        lam_arr = np.broadcast_to(np.asarray(lam, dtype=float), (T,))
        try:
            r = solve_qp(units, T, load, J=J, off=off, elastic=False,
                         enforce_balance=False, lam_offset=lam_arr,
                         tol=max(tol, 1e-8), max_iter=max_iter)
            last_good_P = r.P
            return r.P
        except QPError:
            # 极端 λ 偶发数值失败：沿用上一成功点（λ 已很大时它已逼近可达上限）
            if last_good_P is not None:
                return last_good_P
            # 首次就失败：用爬坡/出力可达上界的一个安全估计（逐时段前向投影到 pmax）
            Px = np.empty((len(units), T))
            init0 = np.array([u["init"] for u in units], dtype=float)
            ru0 = np.array([u["ru"] for u in units], dtype=float)
            rd0 = np.array([u["rd"] for u in units], dtype=float)
            if ru0.ndim == 1:
                ru0 = np.repeat(ru0[:, None], T, 1); rd0 = np.repeat(rd0[:, None], T, 1)
            pmin0 = np.array([u["pmin"] for u in units], dtype=float)
            prev = init0
            for t in range(T):
                Px[:, t] = np.clip(pmax, np.maximum(pmin0, prev - rd0[:, t]),
                                   np.minimum(pmax, prev + ru0[:, t]))
                prev = Px[:, t]
            last_good_P = Px
            return Px

    # 逐时段独立二分 λ_t：发电随 λ 单调增；爬坡耦合下用统一上限仍给出可达发电上界。
    lo_lam = np.zeros(T)
    hi_lam = np.full(T, Mpen)
    P = None
    for it in range(45):
        mid_lam = 0.5 * (lo_lam + hi_lam)
        P = dispatch(mid_lam)
        gen = np.sum(J * P, axis=0)
        deficit = target - gen
        # gen 随 λ 单调增：发电不足(deficit>0)抬高 λ 下界，已够则压低上界
        lo_lam = np.where(deficit > 1e-8, mid_lam, lo_lam)
        hi_lam = np.where(deficit <= 1e-8, mid_lam, hi_lam)
        if on_progress is not None:
            on_progress(it)
        if np.max(hi_lam - lo_lam) < 1e-6:
            break

    lam = 0.5 * (lo_lam + hi_lam)
    P = dispatch(lam)
    gen = np.sum(J * P, axis=0)
    v = np.maximum(target - gen, 0.0)
    feasible = bool(np.all(v <= 1e-4))

    if feasible:
        # 可行：解真正的等微增经济调度（带硬平衡），用二分收敛的逐时段 λ 与对应出力
        # 做 warm start（已接近平衡且爬坡可行），显著改善大负荷摆动算例的收敛。
        def _hard(attempt_tol, use_warm):
            wm = {"P": P.copy(), "y": np.asarray(lam, float)} if use_warm else None
            return solve_qp(units, T, load, J=J, off=off, elastic=False,
                            tol=attempt_tol, max_iter=max_iter, warm=wm)
        for atol, uw in [(max(tol, 1e-8), True), (1e-8, False), (1e-7, False)]:
            try:
                res = _hard(atol, uw)
                break
            except QPError:
                res = None
                continue
        if res is None:
            raise QPError("feasible dispatch failed to converge after retries")
        res.feasible = True
        res.v = np.zeros(T)
        res.M = Mpen
        return res

    # 不可行：P 已在 λ 上限下顶到可达极值，v 即各时段缺口；构造结果。
    res = QPResult(P=P, lam=lam,
                   z_ru=np.zeros((len(units), T)), z_rd=np.zeros((len(units), T)),
                   z_hi=np.zeros((len(units), T)), z_lo=np.zeros((len(units), T)),
                   v=v, iterations=0, gap=0.0, rpri=float(np.max(v)), rdual=0.0)
    res._pmin = np.array([u["pmin"] for u in units], dtype=float)
    res._pmax = pmax
    init = np.array([u["init"] for u in units], dtype=float)
    ru = np.array([u["ru"] for u in units], dtype=float)
    rd = np.array([u["rd"] for u in units], dtype=float)
    if ru.ndim == 1:
        ru = np.repeat(ru[:, None], T, axis=1); rd = np.repeat(rd[:, None], T, axis=1)
    ts_ru = np.empty_like(P); ts_rd = np.empty_like(P)
    ts_ru[:, 0] = init + ru[:, 0] - P[:, 0]
    ts_rd[:, 0] = rd[:, 0] - init + P[:, 0]
    ts_ru[:, 1:] = ru[:, 1:] - (P[:, 1:] - P[:, :-1])
    ts_rd[:, 1:] = rd[:, 1:] + (P[:, 1:] - P[:, :-1])
    res._s_ru, res._s_rd = ts_ru, ts_rd
    res.feasible = False
    res.M = Mpen
    return res


# ---------------- 真实松弛 / 残差 ----------------

def _true_slacks(P, v, d: _QPData):
    """h-Gx：每个不等式约束的真实松弛（可负，表示违反）。"""
    s_lo = P - d.pmin[:, None]
    s_hi = d.pmax[:, None] - P
    s_ru = np.empty_like(P)
    s_rd = np.empty_like(P)
    s_ru[:, 0] = d.init + d.ru[:, 0] - P[:, 0]
    s_rd[:, 0] = d.rd[:, 0] - d.init + P[:, 0]
    s_ru[:, 1:] = d.ru[:, 1:] - (P[:, 1:] - P[:, :-1])
    s_rd[:, 1:] = d.rd[:, 1:] + (P[:, 1:] - P[:, :-1])
    if d.elastic:
        # 弹性建模为等式松弛: ΣJP + v = load+off；v 本身带 0≤v≤vmax
        s_e = None
        s_vlo = v.copy()
        s_vhi = d.vmax - v
    else:
        s_e = s_vlo = s_vhi = None
    return s_lo, s_hi, s_ru, s_rd, s_e, s_vlo, s_vhi


def _mu(s_lo, s_hi, s_ru, s_rd, s_e, s_vlo, s_vhi,
        z_lo, z_hi, z_ru, z_rd, z_e, z_vlo, z_vhi):
    num = float(np.sum(s_lo * z_lo) + np.sum(s_hi * z_hi)
                + np.sum(s_ru * z_ru) + np.sum(s_rd * z_rd))
    m = s_lo.size * 4
    if s_vlo is not None:
        num += float(np.sum(s_vlo * z_vlo) + np.sum(s_vhi * z_vhi))
        m += 2 * s_vlo.size
    return num / max(m, 1)


def _mu_after(ap, ad, s_lo, s_hi, s_ru, s_rd, s_e, s_vlo, s_vhi,
              z_lo, z_hi, z_ru, z_rd, z_e, z_vlo, z_vhi, d):
    def comp(s, z, nm_s, nm_z):
        return float(np.sum((s + ap * d[nm_s]) * np.maximum(z + ad * d[nm_z], 0.0)))

    num = (comp(s_lo, z_lo, "s_lo", "z_lo")
           + comp(s_hi, z_hi, "s_hi", "z_hi")
           + comp(s_ru, z_ru, "s_ru", "z_ru")
           + comp(s_rd, z_rd, "s_rd", "z_rd"))
    m = s_lo.size * 4
    if s_vlo is not None:
        num += (comp(s_vlo, z_vlo, "s_vlo", "z_vlo")
                + comp(s_vhi, z_vhi, "s_vhi", "z_vhi"))
        m += 2 * s_vlo.size
    return num / m


def _dual_contrib(z_lo, z_hi, z_ru, z_rd, z_e=None):
    """平稳条件 Hx+c - A'λ + G'z = 0 中出力上下界与爬坡的 G'z（z_e 已弃用）。"""
    g = -z_lo + z_hi + z_ru - z_rd
    g[:, :-1] += -z_ru[:, 1:] + z_rd[:, 1:]
    return g


def _residuals(d: _QPData, P, v, y,
               s_lo, s_hi, s_ru, s_rd, s_e, s_vlo, s_vhi,
               z_lo, z_hi, z_ru, z_rd, z_e, z_vlo, z_vhi):
    t = _true_slacks(P, v, d)
    stored = [s_lo, s_hi, s_ru, s_rd]
    true_list = [t[0], t[1], t[2], t[3]]
    if d.elastic:
        stored += [s_vlo, s_vhi]
        true_list += [t[5], t[6]]
    rs = max(float(np.max(np.abs(s - ts))) for s, ts in zip(stored, true_list))
    # 平衡：非弹性 J'P=load+off；弹性 J'P+v=load+off；无平衡子问题则不检查
    if d.enforce_balance:
        lhs = np.sum(d.J * P, axis=0) + (v if d.elastic else 0.0)
        r_bal = float(np.max(np.abs(lhs - (d.load + d.off))))
    else:
        r_bal = 0.0

    gz = _dual_contrib(z_lo, z_hi, z_ru, z_rd, None)
    bv = d.b if d.b.ndim == 2 else d.b[:, None]
    rL = bv + 2.0 * d.c[:, None] * P - d.J * y[None, :] + gz
    rD = float(np.max(np.abs(rL)))
    if d.elastic:
        rLv = d.M - y - z_vlo + z_vhi
        rD = max(rD, float(np.max(np.abs(rLv))))
    gap = (float(np.sum(s_lo * z_lo) + np.sum(s_hi * z_hi)
                 + np.sum(s_ru * z_ru) + np.sum(s_rd * z_rd))
           + (float(np.sum(s_vlo * z_vlo) + np.sum(s_vhi * z_vhi))
              if d.elastic else 0.0)) / _nrows(d)
    return max(rs, r_bal, rD), rD, gap, r_bal


def _nrows(d: _QPData) -> int:
    return d.N * d.T * 4 + (2 * d.T if d.elastic else 0)


# ---------------- 牛顿方向 ----------------

def _direction(d: _QPData, P, v, y,
               s_lo, s_hi, s_ru, s_rd, s_e, s_vlo, s_vhi,
               z_lo, z_hi, z_ru, z_rd, z_e, z_vlo, z_vhi,
               corrector, enforce_balance: bool = True):
    """组装并求解一个牛顿方向（仿射 corrector=None；预测-校正 corrector=(σμ,aff)）。"""
    N, T = d.N, d.T
    true = _true_slacks(P, v, d)
    t_lo, t_hi, t_ru, t_rd, _t_e, t_vlo, t_vhi = true
    # slack 残差 r_s = s - (h-Gx)
    q_lo = s_lo - t_lo
    q_hi = s_hi - t_hi
    q_ru = s_ru - t_ru
    q_rd = s_rd - t_rd
    if d.elastic:
        q_vlo = s_vlo - t_vlo
        q_vhi = s_vhi - t_vhi

    def row_terms(s, z, q, ds_aff, dz_aff):
        """w=z/s；仿射强迫 f=z-w·q；校正再加 (ds·dz-σμ)/s。返回 (w,f)。"""
        s_safe = np.maximum(s, 1e-12)
        w = np.minimum(z / s_safe, 1e14)
        if corrector is None:
            f = z - w * q
        else:
            sigma_mu, aff = corrector
            corr = np.clip((aff[ds_aff] * aff[dz_aff] - sigma_mu) / s_safe, -1e14, 1e14)
            f = z - w * q + corr
        f = np.clip(f, -1e14, 1e14)
        return w, f

    w_lo, f_lo = row_terms(s_lo, z_lo, q_lo, "s_lo", "z_lo")
    w_hi, f_hi = row_terms(s_hi, z_hi, q_hi, "s_hi", "z_hi")
    w_ru, f_ru = row_terms(s_ru, z_ru, q_ru, "s_ru", "z_ru")
    w_rd, f_rd = row_terms(s_rd, z_rd, q_rd, "s_rd", "z_rd")
    if d.elastic:
        w_vlo, f_vlo = row_terms(s_vlo, z_vlo, q_vlo, "s_vlo", "z_vlo")
        w_vhi, f_vhi = row_terms(s_vhi, z_vhi, q_vhi, "s_vhi", "z_vhi")

    # ---- 每台机三对角 K_i = H + G'WG ----
    diag = 2.0 * d.c[:, None] + w_lo + w_hi + w_ru + w_rd
    diag[:, :-1] += w_ru[:, 1:] + w_rd[:, 1:]
    # t↔t+1 交叉项：上爬坡 G_t=-1,G_{t+1}=+1 → -w_ru；
    #              下爬坡 G_t=+1,G_{t+1}=-1 → -w_rd
    lower = np.zeros((N, T))
    upper = np.zeros((N, T))
    lower[:, 1:] = -w_ru[:, 1:] - w_rd[:, 1:]
    upper[:, :-1] = lower[:, 1:]

    gz = _dual_contrib(z_lo, z_hi, z_ru, z_rd, z_e)
    # 弹性行现在是等式 ΣJP+v=load+off（v 为非负松弛），不产生 G'z 项
    gz = _dual_contrib(z_lo, z_hi, z_ru, z_rd, None)
    bv = d.b if d.b.ndim == 2 else d.b[:, None]
    rL = bv + 2.0 * d.c[:, None] * P - d.J * y[None, :] + gz
    # fx = -G'f，系数与 _dual_contrib 相反：
    fx = f_lo - f_hi - f_ru + f_rd
    fx[:, :-1] += f_ru[:, 1:] - f_rd[:, 1:]
    rp_P = -(rL + fx)

    _wp = _thomas_solve_batch(lower.copy(), diag, upper, rp_P)
    if _wp.ndim == 3:
        _wp = _wp[..., 0]
    wp = np.asarray(_wp).reshape(N, T)
    ktt = _inv_diag_tridiag(lower, diag, upper)

    # 平衡残差：弹性时等式为 ΣJP+v=load+off，须减去当前 v
    r_eq = d.load + d.off - np.sum(d.J * P, axis=0) - (v if d.elastic else 0.0)
    e_t = np.sum(d.J * wp, axis=0)
    alpha = np.sum(d.J ** 2 * ktt, axis=0)
    beta = np.sum(d.J * ktt, axis=0)
    gamma = np.sum(ktt, axis=0)
    p1 = np.sum(wp, axis=0)

    if d.elastic:
        # 平衡等式 ΣJ_it P_it + v = load+off；dP = wp + K^-1 J dy。
        # v 平稳: M - λ - z_vlo + z_vhi = 0。v 下界 s_vlo=v、上界 s_vhi=vmax-v。
        # 每时段 2×2 (dy,dv)：
        #   α dy + dv = r_eq - e_t
        #   -dy - gv dv = rv,  gv=w_vlo+w_vhi, rv=-(M-y-z_vlo+z_vhi + (-f_vlo+f_vhi))
        gv = w_vlo + w_vhi
        rLv = d.M - y - z_vlo + z_vhi
        fv = -f_vlo + f_vhi
        rv = -(rLv + fv)
        a11 = alpha
        r1 = r_eq - e_t
        r2 = rv
        mm = np.empty((T, 2, 2))
        mm[:, 0, 0] = a11
        mm[:, 0, 1] = 1.0
        mm[:, 1, 0] = -1.0
        mm[:, 1, 1] = -gv
        rr = np.column_stack([np.atleast_1d(r1), np.atleast_1d(r2)])[..., None]
        sol2 = np.linalg.solve(mm, rr)[:, :, 0]
        dy = sol2[:, 0]
        dv = sol2[:, 1]

        dP = wp + _thomas_solve_batch(lower.copy(), diag, upper, d.J * dy[None, :])
    else:
        if not enforce_balance:
            # 无平衡子问题（对偶二分用）：dy=0，P 步直接取 wp（已含线性成本偏移）
            dy = np.zeros(T)
            dv = None
            ds_e = None
            dP = wp
        else:
            dy = (r_eq - e_t) / alpha
            ds_e = None
            dv = None
            dP = wp + _thomas_solve_batch(lower.copy(), diag, upper, d.J * dy[None, :])

    # ---- 存储 slack 与对偶方向 ----
    # 统一约定 s = h - Gx，Δs = -GΔx - q（q=s-(h-Gx) 为平移残差）。
    # 下界 pmin-P≤0: G=-1；上界 P-pmax≤0: G=+1（s_hi=pmax-P）。
    ds_lo = dP - q_lo
    ds_hi = -dP - q_hi
    ds_ru = np.empty_like(dP)
    ds_rd = np.empty_like(dP)
    # 上爬坡 t0: init-P0≤ru → P0-init≤ru，s_ru=init+ru-P0, G_0=-1
    #   t≥1: s_ru=ru-(P_t-P_{t-1})，G_t=+1, G_{t-1}=-1
    # 下爬坡 t0: s_rd=rd-init+P0, G_0=+1
    #   t≥1: s_rd=rd+(P_t-P_{t-1})，G_t=-1, G_{t-1}=+1
    ds_ru[:, 0] = -dP[:, 0] - q_ru[:, 0]
    ds_rd[:, 0] = dP[:, 0] - q_rd[:, 0]
    ds_ru[:, 1:] = -(dP[:, 1:] - dP[:, :-1]) - q_ru[:, 1:]
    ds_rd[:, 1:] = (dP[:, 1:] - dP[:, :-1]) - q_rd[:, 1:]

    # 互补步: Δz = -f - w·Δs
    def dz(f, w, ds):
        return -f - w * ds

    out = {"P": dP, "y": dy,
           "s_lo": ds_lo, "s_hi": ds_hi, "s_ru": ds_ru, "s_rd": ds_rd,
           "z_lo": dz(f_lo, w_lo, ds_lo),
           "z_hi": dz(f_hi, w_hi, ds_hi),
           "z_ru": dz(f_ru, w_ru, ds_ru),
           "z_rd": dz(f_rd, w_rd, ds_rd)}

    if d.elastic:
        # v 边界: s_vlo=v → ds_vlo = dv - q；s_vhi=vmax-v → ds_vhi = -dv - q
        ds_vlo = dv - q_vlo
        ds_vhi = -dv - q_vhi
        out.update({"v": dv, "s_vlo": ds_vlo, "s_vhi": ds_vhi,
                    "z_vlo": dz(f_vlo, w_vlo, ds_vlo),
                    "z_vhi": dz(f_vhi, w_vhi, ds_vhi)})
    return out


def _step_lengths(d: _QPData, dirs,
                  s_lo, s_hi, s_ru, s_rd, s_e, s_vlo, s_vhi,
                  z_lo, z_hi, z_ru, z_rd, z_e, z_vlo, z_vhi,
                  eta=0.99995):
    def max_alpha(var, dvar):
        neg = dvar < -1e-300
        if not np.any(neg):
            return 1.0
        return float(np.min(-var[neg] / dvar[neg]))

    ap = min(max_alpha(s_lo, dirs["s_lo"]), max_alpha(s_hi, dirs["s_hi"]),
             max_alpha(s_ru, dirs["s_ru"]), max_alpha(s_rd, dirs["s_rd"]))
    ad = min(max_alpha(z_lo, dirs["z_lo"]), max_alpha(z_hi, dirs["z_hi"]),
             max_alpha(z_ru, dirs["z_ru"]), max_alpha(z_rd, dirs["z_rd"]))
    if d.elastic:
        ap = min(ap, max_alpha(s_vlo, dirs["s_vlo"]),
                 max_alpha(s_vhi, dirs["s_vhi"]))
        ad = min(ad, max_alpha(z_vlo, dirs["z_vlo"]),
                 max_alpha(z_vhi, dirs["z_vhi"]))
    return min(1.0, eta * ap), min(1.0, eta * ad)
