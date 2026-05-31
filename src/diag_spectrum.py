"""诊断偏蓝：画绿松石带高度的出射光谱，对照 Chappuis，看是不是绿黄端被多削。

文献(交叉验证): 偏蓝不能甩给瑞利(τ@450,25km≈0.44 量级对)。
更可能 (1)绿黄端(500-560nm)被 Chappuis 削太狠, 或 (2)蓝端残留权重高, 或 (3)色彩转换。
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import radiative_transfer as rt
import cross_sections as cs
import geometry, atmosphere, solar, color as col

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
lam = np.linspace(380, 720, 341)

# 几个代表擦边高度
for h in [25.0]:
    N_air = geometry.column_density(h, atmosphere.n_air)
    N_o3 = geometry.column_density(h, atmosphere.n_o3)
    tau_ray = N_air * cs.sigma_rayleigh(lam)
    tau_o3 = N_o3 * cs.sigma_o3(lam)
    T = np.exp(-(tau_ray + tau_o3))
    I_sun = solar.solar_spectrum(lam)
    I_out = I_sun * T
    I_out_n = I_out / I_out.max()

    fig, ax = plt.subplots(2, 1, figsize=(9, 8))
    ax[0].plot(lam, np.exp(-tau_ray), label="瑞利透射 exp(-τ_R)", color="steelblue")
    ax[0].plot(lam, np.exp(-tau_o3), label="臭氧透射 exp(-τ_O3)", color="green")
    ax[0].plot(lam, T, label="总透射 T", color="k", lw=2)
    ax[0].set_title(f"h={h:.0f}km 透射率分解（绿松石带高度）")
    ax[0].set_ylabel("透射率"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].axvspan(500, 560, alpha=0.1, color="green")  # 绿黄端
    ax[0].axvspan(440, 480, alpha=0.1, color="blue")   # 蓝端

    ax[1].plot(lam, I_out_n, color="purple", lw=2, label="出射谱(归一)")
    ax[1].set_title("出射光谱：峰在哪决定色相")
    ax[1].set_xlabel("波长 (nm)"); ax[1].set_ylabel("相对强度"); ax[1].grid(alpha=0.3)
    peak = lam[np.argmax(I_out)]
    ax[1].axvline(peak, color="red", ls="--", label=f"峰 {peak:.0f}nm")
    ax[1].legend()
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC"]
    plt.rcParams["axes.unicode_minus"] = False
    fig.tight_layout()
    p = os.path.join(OUT, "diag_spectrum_25km.png")
    fig.savefig(p, dpi=130)
    print(f"已存 {p}")

    # 数值诊断
    print(f"\nh={h}km 出射谱峰波长: {peak:.0f}nm")
    for w in [440, 470, 500, 530, 560, 600, 660]:
        j = np.argmin(np.abs(lam - w))
        print(f"  {w}nm: 瑞利T={np.exp(-tau_ray[j]):.3f} 臭氧T={np.exp(-tau_o3[j]):.3f} 总T={T[j]:.3f} 出射={I_out_n[j]:.3f}")
    XYZ = col.spectrum_to_XYZ(lam, I_out)
    white = col.spectrum_to_XYZ(lam, I_sun)
    print(f"\n色相角(对白点): {col.hue_angle(XYZ, white_XYZ=white):.1f}° (文献: 微偏蓝, teal~190-220°)")
    # 蓝/绿黄强度比
    b = I_out_n[np.argmin(np.abs(lam-460))]
    gy = I_out_n[np.argmin(np.abs(lam-530))]
    r = I_out_n[np.argmin(np.abs(lam-660))]
    print(f"蓝(460)={b:.3f} 绿黄(530)={gy:.3f} 红(660)={r:.3f}; 蓝/绿黄={b/gy:.2f}")
