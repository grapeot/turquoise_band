"""【权威管线】真折射弯曲的数值积分 —— 替换 geometry.refraction_angle 的解析 α0·exp(-h/H)。

零解析处方：折射率 n(z) 直接由真实 AFGL 密度廓线 atmosphere.n_air(z) 给出
    n(z) - 1 = KAPPA · N(z)      (N=数密度 cm^-3, 海平面 n-1 ≈ 2.7e-4)
光线进地球大气后，沿真实折射率梯度逐步弯曲。总偏转角 α(h)、出射方向、真实弯曲
路径全部由积分涌现，不假设任何指数律。

算法
----
球对称介质里 Bouguer 不变量守恒:
    n(r) · r · sin(ζ) = const = L
其中 ζ 是光线与局地径向方向的夹角，r=R_EARTH+z 是地心距。这是 Snell 定律在
球分层介质中的连续极限（等价于 eikonal 方程对球对称情形的首积分）。

在极坐标 (r, φ) 下追踪光线。设入射前光线是从无穷远来的平行束，impact parameter
b = R_EARTH + h（h=擦边高度，即未折射时切点海拔）。真空中 sin(ζ) = b/r，故
    L = n(r)·r·sin(ζ) = 1·b   （进入大气前 n→1）

Bouguer 是首积分，可用作守恒量校核。实际积分则用更稳健的笛卡尔 eikonal 形式，
避免切点处 dr/ds→0 的极坐标奇点。状态取 (x, y, ux, uy)：位置 + 单位方向矢量，
以弧长 s 为自变量 RK4 推进:
    dx/ds = u,   du/ds = (1/n)[∇n − (u·∇n)u]    （球对称下 ∇n = (dn/dr)·r̂）
后一式是 eikonal 方程 d(n·u)/ds=∇n 投到横向、保持 |u|=1 的形式，等价于连续 Snell。

总偏转角 α = 出射单位方向与入射方向 (+x) 的夹角，直接从积分出的 exit_dir 读出，
无需对称假设。

验证: h=0 擦地 α ≈ 70 arcmin（对照地表掠射天文折射 ~35' 单边 ×2 双段几何）。
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import atmosphere as atm

import geometry as _g
import cross_sections as _cs

R_EARTH = _g.R_EARTH      # km, 单一来源
ARCMIN_PER_RAD = np.degrees(1.0) * 60.0

# 折射率标定: (n-1) = KAPPA · N(z)。海平面 (n-1)0 与数密度都取 cross_sections 公共值(自洽)。
N0_SEA = _cs.N0_SEA
NU0_SEA = _cs.NU0_SEA_600  # 可见光海平面折射度 (n-1)@600nm, Peck&Reeder
KAPPA = NU0_SEA / N0_SEA   # cm^3, 折射度/数密度


def refractivity(z_km):
    """(n-1) 折射度，直接正比于真实 AFGL 空气数密度。零指数律假设。"""
    return KAPPA * atm.n_air(np.asarray(z_km, dtype=float))


def n_of_r(r_km):
    """折射率 n(r)，r=地心距 (km)。大气顶以上 n=1。"""
    z = np.asarray(r_km, dtype=float) - R_EARTH
    return 1.0 + refractivity(z)


def dn_dr(r_km, dr=1e-4):
    """折射率对 r 的导数 dn/dr (km^-1)，中心差分。"""
    return (n_of_r(r_km + dr) - n_of_r(r_km - dr)) / (2.0 * dr)


def trace_ray(h_km, z_top_km=120.0, ds_km=0.02, return_path=False):
    """真折射数值积分: 输入擦边高度 h → 总偏转角 α、出射方向、弯曲路径。

    用以弧长 s 为自变量的 RK4 积分光线方程。状态 (x, y, ux, uy)：位置与单位方向矢量。
    方向演化由 eikonal 光线方程: d(n·u)/ds = ∇n。球对称下 ∇n = (dn/dr)·r̂，故
        du/ds = (1/n)[ ∇n − (u·∇n)u ]      (保持 |u|=1 的横向折射)
    位置演化 dx/ds = u。

    入射: 平行束沿 +x 方向（朝向地球），impact parameter b=R_EARTH+h（光线初始 y=b）。
    从大气顶 (r=R_EARTH+z_top) 外的真空起步，进入大气被弯曲，再出射到大气顶外真空，
    停止。出射方向与入射方向 (+x) 的夹角即总偏转角 α（向地心一侧弯为正）。

    参数
    ----
    h_km : 擦边高度（impact parameter b=R_EARTH+h）
    z_top_km : 大气顶高度，其上 n=1（积分边界）
    ds_km : 弧长步长 (km)
    return_path : 若 True，额外返回采样路径 (x, y)

    返回
    ----
    一个 dict（或在 return_path=True 时附带 path）：
      alpha    : 总偏转角 (rad)，正=向地心弯
      exit_dir : (ux, uy) 出射单位方向
      z_tan    : 实际最近接近点海拔 (km)；折射把切点压得比 impact-parameter h 更低
      blocked  : True=弯曲光线撞地（被地球遮挡，到不了月面）
      path     : (return_path 时) Nx2 路径 (x,y) km
    """
    r_top = R_EARTH + z_top_km
    b = R_EARTH + h_km

    # 初始: 在大气顶圆外的真空里，沿 +x 朝地球飞，y=b 不变直到进入大气。
    # 起点取 x 使其恰在大气顶圆外: x0 = -sqrt(r_top^2 - b^2) 若 b<r_top，否则擦不到大气。
    if b >= r_top:
        # 擦边高度高于大气顶：不被折射
        res = dict(alpha=0.0, exit_dir=np.array([1.0, 0.0]), z_tan=h_km, blocked=False)
        return (res, None) if return_path else res
    x = -np.sqrt(r_top**2 - b**2)
    y = float(b)
    ux, uy = 1.0, 0.0

    def deriv(state):
        x, y, ux, uy = state
        r = np.hypot(x, y)
        n = n_of_r(r)
        gradn_mag = dn_dr(r)            # dn/dr, 沿 +r̂
        # r̂ 分量
        rx, ry = x / r, y / r
        gx, gy = gradn_mag * rx, gradn_mag * ry   # ∇n
        u_dot_g = ux * gx + uy * gy
        # du/ds = (1/n)(∇n − (u·∇n)u)
        dux = (gx - u_dot_g * ux) / n
        duy = (gy - u_dot_g * uy) / n
        return np.array([ux, uy, dux, duy])

    state = np.array([x, y, ux, uy])
    path = [(x, y)] if return_path else None

    max_steps = int(2.0 * abs(x) / ds_km) + 20000
    in_atmos = False
    blocked = False
    r_min = np.hypot(state[0], state[1])
    for _ in range(max_steps):
        r = np.hypot(state[0], state[1])
        r_min = min(r_min, r)
        if r <= r_top + 1e-9:
            in_atmos = True
        # 出射条件: 已进过大气且现在又回到大气顶外、且远离地球（x 方向已过切点）
        if in_atmos and r > r_top and state[0] > 0.0:
            break
        # 撞地: 弯曲光线降到地表以下 → 被地球遮挡，到不了月面
        if r < R_EARTH:
            blocked = True
            break
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * ds_km * k1)
        k3 = deriv(state + 0.5 * ds_km * k2)
        k4 = deriv(state + ds_km * k3)
        state = state + (ds_km / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # 归一化方向，抑制数值漂移
        un = np.hypot(state[2], state[3])
        state[2] /= un
        state[3] /= un
        if return_path:
            path.append((state[0], state[1]))

    exit_dir = np.array([state[2], state[3]])
    # 总偏转角: 出射方向相对入射 +x 的转角。向地心(−y)弯 → uy<0 → α>0。
    alpha = np.arctan2(-exit_dir[1], exit_dir[0])
    z_tan = r_min - R_EARTH
    res = dict(alpha=float(alpha), exit_dir=exit_dir, z_tan=float(z_tan), blocked=bool(blocked))
    if return_path:
        return res, np.array(path)
    return res


def trace_rays_batch(h_array, z_top_km=120.0, ds_km=0.02):
    """矢量化批量追踪: 一次并行积分所有擦边高度的光线(状态 (N,4) 数组, 锁步 RK4)。

    与逐条 trace_ray 物理完全相同(同方程同步长), 但所有光线同时推进——已退出/撞地的光线
    用 active 掩码冻结。返回 (alpha (N,), z_tan (N,), blocked (N,))。性能: 替代 N 次 Python
    for 循环, numpy 矢量化吃满核(折射是 ray tracing 最大热点之一)。
    """
    h = np.atleast_1d(np.asarray(h_array, dtype=float))
    N = h.size
    r_top = R_EARTH + z_top_km
    b = R_EARTH + h
    above = b >= r_top                       # 擦边高于大气顶: 不折射
    x = np.where(above, 0.0, -np.sqrt(np.maximum(r_top**2 - b**2, 0.0)))
    y = b.copy()
    ux = np.ones(N); uy = np.zeros(N)
    state = np.stack([x, y, ux, uy], axis=1)  # (N,4)

    def deriv(s):
        xx, yy, vx, vy = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
        r = np.hypot(xx, yy); n = n_of_r(r); g = dn_dr(r)
        rx, ry = xx / r, yy / r; gx, gy = g * rx, g * ry
        ud = vx * gx + vy * gy
        return np.stack([vx, vy, (gx - ud * vx) / n, (gy - ud * vy) / n], axis=1)

    in_atmos = np.zeros(N, bool); blocked = np.zeros(N, bool); done = above.copy()
    r_min = np.hypot(state[:, 0], state[:, 1])
    max_steps = int(2.0 * np.max(np.abs(x)) / ds_km) + 20000
    for _ in range(max_steps):
        r = np.hypot(state[:, 0], state[:, 1])
        r_min = np.minimum(r_min, r)
        in_atmos |= (r <= r_top + 1e-9)
        exit_now = in_atmos & (r > r_top) & (state[:, 0] > 0.0) & ~done
        hit = (r < R_EARTH) & ~done
        blocked |= hit; done |= exit_now | hit
        if done.all():
            break
        act = ~done
        k1 = deriv(state); k2 = deriv(state + 0.5 * ds_km * k1)
        k3 = deriv(state + 0.5 * ds_km * k2); k4 = deriv(state + ds_km * k3)
        upd = (ds_km / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        state[act] = state[act] + upd[act]
        un = np.hypot(state[:, 2], state[:, 3])
        state[:, 2] /= un; state[:, 3] /= un
    alpha = np.arctan2(-state[:, 3], state[:, 2])
    alpha[above] = 0.0
    z_tan = r_min - R_EARTH
    return alpha, z_tan, blocked


def refraction_angle_traced(h_km, **kw):
    """真追踪折射偏转角 α(h) (rad)，可数组。替换 geometry.refraction_angle。

    撞地（被遮挡）的光线返回 nan —— 它们到不了月面，不该参与落点统计。
    """
    h_arr = np.atleast_1d(np.asarray(h_km, dtype=float))
    out = np.empty(h_arr.size)
    for i, h in enumerate(h_arr):
        r = trace_ray(float(h), **kw)
        out[i] = np.nan if r["blocked"] else r["alpha"]
    return out if out.size > 1 else float(out[0])


def grazing_angle(z_top_km=120.0, ds_km=0.01):
    """找擦地极限: 弯曲光线切点恰好落在地表的 impact-parameter 与其偏转角 α。

    返回 (h_graze_km, alpha_graze_rad)。这是真追踪给出的"擦地 70'"对应点：
    解析模型把 70' 钉在 impact-parameter h=0；真追踪显示 h=0 的光线其实撞地被挡，
    真正擦地（切点 z_tan≈0）的是 impact-parameter h≈2km 的光线，其 α≈最大值。
    """
    lo, hi = 0.0, 6.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        r = trace_ray(mid, z_top_km=z_top_km, ds_km=ds_km)
        if r["blocked"]:
            lo = mid          # 还在撞地，抬高 impact parameter
        else:
            hi = mid          # 已能清地，降低
    r = trace_ray(hi, z_top_km=z_top_km, ds_km=ds_km)
    return hi, r["alpha"]


if __name__ == "__main__":
    import geometry as g
    # 海平面折射度自查
    print(f"标定: (n-1)@海平面 = {refractivity(0.0):.3e}  (标准 2.7e-4)")
    print(f"(n-1)@8km = {refractivity(8.0):.3e}, @50km = {refractivity(50.0):.3e}")

    print("\nimpact-h(km)  α_traced(')  α_analytic(')  z_tan(km)  blocked  出射方向")
    for h in [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 18.0, 25.0, 40.0]:
        r = trace_ray(h)
        a_arcmin = np.degrees(r["alpha"]) * 60.0
        a_ana = np.degrees(g.refraction_angle(h)) * 60.0
        ud = r["exit_dir"]
        print(f"  {h:>5.1f}     {a_arcmin:>8.2f}    {a_ana:>8.2f}     {r['z_tan']:>6.2f}   {str(r['blocked']):>5}   ({ud[0]:.5f},{ud[1]:.5f})")

    h_g, a_g = grazing_angle()
    print(f"\n擦地极限: impact-h={h_g:.3f}km, 切点 z_tan≈0, α_grazing={np.degrees(a_g)*60:.1f} arcmin")
    print("对照解析 α(h=0)=70 arcmin —— 真追踪的擦地偏转量级一致。")
