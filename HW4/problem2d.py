"""Problem 2d: Competitive two-population circuit vs drift-diffusion.

    dr1/dt = -r1 + ws r1 - wc r2 + I0 + mu/2 + sigma xi1,
    dr2/dt = -r2 + ws r2 - wc r1 + I0 - mu/2 + sigma xi2,
mu = 0.2, sigma = 0.5, I0 = 5, ws = wc = (1 + lam)/2, lam in {0, 0.2}, r1 = r2 = I0 at t = 0.
Euler-Maruyama, dt = 0.002, each population with its own noise sigma sqrt(dt) N(0,1), 1e4 trials.
Decision: first time |d| = |r1 - r2| >= theta; correct if d > 0. All thresholds theta = 0.5, 1, ..., 4 are
read off the same trials (a path must cross the lower thresholds first).

From part (a), d obeys dd/dt = lam d + mu + sqrt(2) sigma xi.
Predictions:
  (b) lam = 0:  P = 1/(1 + exp(-mu theta/sigma^2)),  <T> = (theta/mu) tanh(mu theta/(2 sigma^2)).
  (c) lam > 0:  P = int_{-theta}^0 g / int_{-theta}^theta g,  g(y) = exp(-(lam y^2 + 2 mu y)/(2 sigma^2)),
                saturating at Phi(mu/(sigma sqrt(lam))) as theta -> infinity.
      <T> for lam > 0 has no closed form; it solves sigma^2 T'' + (lam x + mu) T' = -1, T(+-theta) = 0, i.e.
      (T'/g)' = -1/(sigma^2 g), which is done here by quadrature. The same routine at lam = 0 must reproduce (b).

Euler-Maruyama only checks the threshold every dt, so paths overshoot and the effective boundary sits at about
theta + 0.5826 sqrt(2) sigma sqrt(dt) (Siegmund's correction for a discretely monitored Brownian motion), which
is 0.018 here. It matters only at small theta; it is checked by rerunning theta = 0.5 at dt/4.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.stats import norm

MU, SIG, I0, DT, NTRIAL = 0.2, 0.5, 5.0, 0.002, 10_000
THETAS = np.arange(0.5, 4.01, 0.5)
BETA = 0.5826  # -zeta(1/2)/sqrt(2 pi)


def simulate(lam, thetas, dt=DT, n=NTRIAL, seed=0):
    """Decision times and choices (+1 correct) for every threshold, plus the smallest rate seen."""
    rng = np.random.default_rng(seed)
    w = (1 + lam) / 2
    nth = len(thetas)
    T = np.full((nth, n), np.nan)
    C = np.zeros((nth, n))
    r1, r2 = np.full(n, I0), np.full(n, I0)
    idx = np.arange(n)            # trials still undecided at the largest threshold
    nxt = np.zeros(n, dtype=int)  # index of the next threshold each trial has not yet crossed
    rmin, k = I0, 0
    while idx.size:
        k += 1
        x1, x2 = rng.standard_normal((2, idx.size))
        dr1 = (-r1 + w * r1 - w * r2 + I0 + MU / 2) * dt + SIG * np.sqrt(dt) * x1
        dr2 = (-r2 + w * r2 - w * r1 + I0 - MU / 2) * dt + SIG * np.sqrt(dt) * x2
        r1, r2 = r1 + dr1, r2 + dr2
        rmin = min(rmin, r1.min(), r2.min())
        d = r1 - r2
        for j in range(nth):  # in increasing order, so one step can cross several thresholds
            hit = (nxt == j) & (np.abs(d) >= thetas[j])
            T[j, idx[hit]] = k * dt
            C[j, idx[hit]] = np.sign(d[hit])
            nxt[hit] += 1
        keep = nxt < nth
        if not keep.all():
            idx, r1, r2, nxt = idx[keep], r1[keep], r2[keep], nxt[keep]
    return T, C, rmin


def ddm_theory(theta):
    return 1 / (1 + np.exp(-MU * theta / SIG**2)), theta / MU * np.tanh(MU * theta / (2 * SIG**2))


def backward_theory(lam, theta, nx=20001):
    """P(correct) and <T> starting from d = 0, by quadrature of the backward equations."""
    x = np.linspace(-theta, theta, nx)
    g = np.exp(-(lam * x**2 + 2 * MU * x) / (2 * SIG**2))
    Gc = cumulative_trapezoid(g, x, initial=0)
    i0 = nx // 2
    P = Gc[i0] / Gc[-1]
    # T' = g (A - F),  F(x) = int_{-theta}^x 1/(sigma^2 g);  T(-theta) = 0 and T(theta) = 0 fix A
    F = cumulative_trapezoid(1 / (SIG**2 * g), x, initial=0)
    I_g = Gc
    I_gF = cumulative_trapezoid(g * F, x, initial=0)
    A = I_gF[-1] / I_g[-1]
    T0 = A * I_g[i0] - I_gF[i0]
    return P, T0


def summarize(T, C):
    P = (C > 0).mean(axis=1)
    mT = np.nanmean(T, axis=1)
    return P, np.sqrt(P * (1 - P) / C.shape[1]), mT, np.nanstd(T, axis=1, ddof=1) / np.sqrt(C.shape[1])


if __name__ == "__main__":
    # --- validation of the quadrature against the closed forms of (b) ---
    for th in THETAS:
        Pq, Tq = backward_theory(0.0, th)
        Pc, Tc = ddm_theory(th)
        assert abs(Pq - Pc) < 1e-6 and abs(Tq - Tc) < 1e-6, (th, Pq, Pc, Tq, Tc)
    print("quadrature at lam = 0 reproduces P and <T> of (b) to < 1e-6 at all theta")
    th_fine = np.linspace(0.05, 4.5, 200)

    sims, theory = {}, {}
    for lam in (0.0, 0.2):
        T, C, rmin = simulate(lam, THETAS)
        assert rmin > 0 and not np.isnan(T).any()
        sims[lam] = summarize(T, C)
        theory[lam] = np.array([backward_theory(lam, th) for th in THETAS]).T
        P, seP, mT, seT = sims[lam]
        Pt, Tt = theory[lam]
        print(f"\nlam = {lam:g} (ws = wc = {(1+lam)/2:g}), {NTRIAL} trials, min rate {rmin:.3f} > 0")
        # theory with the boundary moved out by the Euler-Maruyama overshoot
        Tt_eff = np.array([backward_theory(lam, th + BETA * np.sqrt(2 * DT) * SIG)[1] for th in THETAS])
        print(" theta |  P sim  (SE)    P theory   z  |  <T> sim  (SE)    <T> theory    z   | "
              "overshoot-corrected <T>   z")
        for j, th in enumerate(THETAS):
            print(f"  {th:4.1f} | {P[j]:.4f} ({seP[j]:.4f})  {Pt[j]:.4f}  {(P[j]-Pt[j])/seP[j]:+5.2f} | "
                  f"{mT[j]:8.4f} ({seT[j]:.4f})  {Tt[j]:8.4f}  {(mT[j]-Tt[j])/seT[j]:+5.2f} | "
                  f"{Tt_eff[j]:8.4f}  {(mT[j]-Tt_eff[j])/seT[j]:+5.2f}")
    print(f"\nlam = 0.2 saturation Phi(mu/(sigma sqrt(lam))) = {norm.cdf(MU/(SIG*np.sqrt(0.2))):.4f}")

    # --- discretization check at theta = 0.5 ---
    for lam in (0.0, 0.2):
        T4, C4, _ = simulate(lam, THETAS[:1], dt=DT / 4, seed=7)
        P4, seP4, mT4, seT4 = summarize(T4, C4)
        Pt, Tt = backward_theory(lam, THETAS[0])
        print(f"dt check lam = {lam:g}, theta = 0.5: <T> = {sims[lam][2][0]:.4f} at dt = {DT}, "
              f"{mT4[0]:.4f} ({seT4[0]:.4f}) at dt = {DT/4}, theory {Tt:.4f}; "
              f"P = {sims[lam][0][0]:.4f}, {P4[0]:.4f}, theory {Pt:.4f}")

    # --- leaky-competing circuit vs perfect integrator at matched accuracy ---
    P_c, seP_c, T_c, seT_c = (a[-1] for a in sims[0.2])
    th_m = SIG**2 / MU * np.log(P_c / (1 - P_c))
    Pm_th, Tm_th = ddm_theory(th_m)
    Tm, Cm, _ = simulate(0.0, np.array([th_m]), seed=11)
    Pm, sePm, mTm, seTm = summarize(Tm, Cm)
    print(f"\nlam = 0.2, theta = 4:     P = {P_c:.4f} ({seP_c:.4f}), <T> = {T_c:.4f} ({seT_c:.4f})")
    print(f"perfect integrator matched: theta' = (sigma^2/mu) ln(P/(1-P)) = {th_m:.4f}")
    print(f"   theory: P = {Pm_th:.4f}, <T> = {Tm_th:.4f}")
    print(f"   simulated (lam = 0, theta'): P = {Pm[0]:.4f} ({sePm[0]:.4f}), <T> = {mTm[0]:.4f} ({seTm[0]:.4f})")
    print(f"   circuit / integrator decision time = {T_c/mTm[0]:.4f}")

    # --- figure ---
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for lam, col in ((0.0, "k"), (0.2, "tab:red")):
        P, seP, mT, seT = sims[lam]
        Pf, Tf = np.array([backward_theory(lam, th) for th in th_fine]).T
        lab = "(b)" if lam == 0 else "(c)"
        ax[0].plot(th_fine, Pf, color=col, lw=1.5, label=rf"theory {lab}, $\lambda={lam:g}$")
        ax[0].plot(THETAS, P, "o", color=col, ms=5, label=rf"simulation, $\lambda={lam:g}$")
        ax[1].plot(th_fine, Tf, color=col, lw=1.5,
                   label=rf"theory, $\lambda={lam:g}$" + (" (closed form)" if lam == 0 else " (quadrature)"))
        ax[1].plot(THETAS, mT, "o", color=col, ms=5, label=rf"simulation, $\lambda={lam:g}$")
    ax[0].axhline(norm.cdf(MU / (SIG * np.sqrt(0.2))), color="tab:red", ls="--", lw=1,
                  label=r"$\Phi(\mu/\sigma\sqrt{\lambda})=%.4f$" % norm.cdf(MU / (SIG * np.sqrt(0.2))))
    ax[0].plot(th_m, Pm[0], "s", color="tab:blue", ms=7,
               label=rf"perfect integrator at $\theta'={th_m:.3f}$ (matched to $\lambda=0.2,\theta=4$)")
    ax[1].plot(th_m, mTm[0], "s", color="tab:blue", ms=7, label=rf"perfect integrator at $\theta'={th_m:.3f}$")
    for a in ax:
        a.set_xlabel(r"threshold $\theta$ on $|r_1-r_2|$")
        a.legend(fontsize=8)
    ax[0].set_ylabel("P(correct)")
    ax[1].set_ylabel(r"mean decision time $\langle T\rangle$ (units of rate time constant)")
    fig.suptitle(rf"two-population circuit, $\mu={MU}$, $\sigma={SIG}$, $I_0={I0:g}$, $w_s=w_c=(1+\lambda)/2$, "
                 rf"Euler-Maruyama $dt={DT}$, $10^4$ trials per point")
    fig.tight_layout()
    fig.savefig("problem2d_ddm.png", dpi=150)
