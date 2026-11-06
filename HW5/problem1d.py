"""Problem 1d: Wilson-Cowan E-I oscillation just past the Hopf point.

    tau_E dE/dt = -E + f(w_EE E - w_EI I + p_E),   tau_I dI/dt = -I + f(w_IE E + p_I),   f(x) = 1/(1 + e^{-eta x}),
eta = 4, tau_E = 5 ms, w_EI = w_IE = 10, equilibrium held at (E, I) = (0.1, 0.1).

From part (a):  w_EE^Hopf(tau_I) = (1 + tau_E/tau_I) / (eta E(1-E)),
                p_E = -(1/eta) ln((1-E)/E) - w_EE E + w_EI I,   p_I = -(1/eta) ln((1-I)/I) - w_IE E.
p_E is recomputed for every w_EE = (1+eps) w_EE^Hopf so the equilibrium stays at (0.1, 0.1); otherwise the fixed
point (and with it B, omega) would move with eps.
From part (b): omega^2 = B/tau_I - 1/tau_I^2, B = eta^2 w_EI w_IE E(1-E) I(1-I)/tau_E.
From part (c): linear phase lag of I behind E = arctan(omega tau_I).

Simulation: solve_ivp (DOP853, rtol 1e-10) from (E, I) + (1e-3, 0). The linear growth rate is
sigma = eps (1 + tau_E/tau_I)/(2 tau_E), so the run lasts 40/sigma (at least 3000 ms) before measuring over the last
40 periods. Peaks of E and I are found on a 0.01 ms grid and refined by a parabola through the three samples.
Period = mean spacing of E peaks; amplitude = max E - min E; lag = mean (next I peak - E peak) as a fraction of
the period, in degrees.

Validation:
  * the Jacobian at eps = 0, by finite differences of the right-hand side, must have eigenvalues +-i omega(b);
  * "settled" means the amplitude over the first and second halves of the measuring window agree to 1e-3;
  * as eps -> 0 the period and lag must approach 2 pi/omega and arctan(omega tau_I);
  * a supercritical Hopf gives amplitude ~ sqrt(eps), i.e. a log-log slope near 1/2 for small eps.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

ETA, TAU_E, W_EI, W_IE = 4.0, 5.0, 10.0, 10.0
EBAR, IBAR = 0.1, 0.1
OUT = os.path.dirname(os.path.abspath(__file__))


def f(x):
    return 1.0 / (1.0 + np.exp(-ETA * x))


def w_ee_hopf(tau_i):
    return (1 + TAU_E / tau_i) / (ETA * EBAR * (1 - EBAR))


def p_e(w_ee):
    return -np.log((1 - EBAR) / EBAR) / ETA - w_ee * EBAR + W_EI * IBAR


P_I = -np.log((1 - IBAR) / IBAR) / ETA - W_IE * EBAR
B = ETA**2 * W_EI * W_IE * EBAR * (1 - EBAR) * IBAR * (1 - IBAR) / TAU_E


def omega_lin(tau_i):
    return np.sqrt(B / tau_i - 1 / tau_i**2)


def rhs(t, y, w_ee, pe, tau_i):
    E, I = y
    return [(-E + f(w_ee * E - W_EI * I + pe)) / TAU_E, (-I + f(W_IE * E + P_I)) / tau_i]


def jacobian_eigs(tau_i, h=1e-6):
    """Finite-difference Jacobian at (EBAR, IBAR) for w_EE = w_EE^Hopf."""
    w = w_ee_hopf(tau_i)
    args = (w, p_e(w), tau_i)
    y0 = np.array([EBAR, IBAR])
    assert np.allclose(rhs(0, y0, *args), 0, atol=1e-14), "(0.1, 0.1) must be an equilibrium"
    J = np.empty((2, 2))
    for j in range(2):
        dy = np.zeros(2)
        dy[j] = h
        J[:, j] = (np.array(rhs(0, y0 + dy, *args)) - np.array(rhs(0, y0 - dy, *args))) / (2 * h)
    return np.linalg.eigvals(J)


def peak_times(t, x):
    idx, _ = find_peaks(x)
    idx = idx[(idx > 0) & (idx < len(x) - 1)]
    y0, y1, y2 = x[idx - 1], x[idx], x[idx + 1]
    shift = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return t[idx] + shift * (t[1] - t[0])


def run(tau_i, eps, n_meas=40, dt_s=0.01):
    """Simulate and measure the settled oscillation. Returns dict of period, amplitude, lag, and traces."""
    w = (1 + eps) * w_ee_hopf(tau_i)
    args = (w, p_e(w), tau_i)
    sigma = eps * (1 + TAU_E / tau_i) / (2 * TAU_E)
    T_lin = 2 * np.pi / omega_lin(tau_i)
    t_settle = max(3000.0, 40.0 / sigma)
    t_end = t_settle + n_meas * T_lin
    sol = solve_ivp(rhs, (0, t_end), [EBAR + 1e-3, IBAR], args=args, method="DOP853",
                    rtol=1e-10, atol=1e-12, dense_output=True)
    t = np.arange(t_settle, t_end, dt_s)
    E, I = sol.sol(t)
    tE, tI = peak_times(t, E), peak_times(t, I)
    period = np.mean(np.diff(tE))
    half = len(t) // 2
    amp1, amp2 = np.ptp(E[:half]), np.ptp(E[half:])
    assert abs(amp1 - amp2) <= 1e-3 * amp2, f"not settled: tau_I={tau_i}, eps={eps}, {amp1} vs {amp2}"
    lags = []
    for te in tE:
        later = tI[tI > te]
        if len(later):
            lags.append(later[0] - te)
    lag_deg = 360 * np.mean(lags) / period
    return dict(period=period, amp=np.ptp(E), lag=lag_deg, sol=sol, t_end=t_end)


def main():
    # --- validation: Jacobian at the Hopf point ------------------------------------------------------------------
    tau_i = 10.0
    w0 = omega_lin(tau_i)
    ev = jacobian_eigs(tau_i)
    print(f"w_EE^Hopf = {w_ee_hopf(tau_i):.4f}, p_E = {p_e(w_ee_hopf(tau_i)):.4f}, p_I = {P_I:.4f}, B = {B:.4f}")
    print(f"Jacobian eigenvalues at Hopf (tau_I=10): {ev[0]:.6f}, {ev[1]:.6f};  omega(b) = {w0:.6f} rad/ms")
    assert np.allclose(ev.real, 0, atol=1e-8) and np.isclose(abs(ev[0].imag), w0, rtol=1e-6)
    T0, lag0 = 2 * np.pi / w0, np.degrees(np.arctan(w0 * tau_i))
    print(f"Linear predictions: period = {T0:.4f} ms, f = {1000 / T0:.3f} Hz, lag = {lag0:.3f} deg\n")

    # --- (i) eps sweep at tau_I = 10 -----------------------------------------------------------------------------
    eps_list = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    res = [run(tau_i, e) for e in eps_list]
    print("(i) tau_I = 10 ms")
    print(f"{'eps':>6} {'period (ms)':>12} {'f (Hz)':>9} {'amp maxE-minE':>14} {'lag (deg)':>10}")
    for e, r in zip(eps_list, res):
        print(f"{e:6.3f} {r['period']:12.4f} {1000 / r['period']:9.3f} {r['amp']:14.5f} {r['lag']:10.3f}")
    amps = np.array([r["amp"] for r in res])
    slope_all, icpt_all = np.polyfit(np.log(eps_list), np.log(amps), 1)
    slope_small, icpt_small = np.polyfit(np.log(eps_list[:3]), np.log(amps[:3]), 1)
    print(f"log-log slope: all eps = {slope_all:.4f};  eps <= 0.02 = {slope_small:.4f}  (supercritical Hopf: 0.5)")
    # Hopf normal form: period shift ~ amp^2 ~ eps, peak-timing (lag) shift ~ amp ~ sqrt(eps). Extrapolate to eps=0.
    per = np.array([r["period"] for r in res])
    lag = np.array([r["lag"] for r in res])
    per0 = np.polyfit(eps_list[:3], per[:3], 1)[1]
    lag0_fit = np.polyfit(np.sqrt(eps_list[:3]), lag[:3], 1)[1]
    print(f"eps->0 extrapolation (3 smallest eps): period {per0:.4f} ms (linear {T0:.4f}), "
          f"lag {lag0_fit:.3f} deg (linear {lag0:.3f})\n")

    # --- (ii) tau_I sweep at eps = 0.02 --------------------------------------------------------------------------
    taus = np.array([2, 3, 5, 7.5, 10, 15, 20])
    print("(ii) eps = 0.02")
    print(f"{'tau_I':>6} {'f_sim (Hz)':>11} {'omega/2pi (Hz)':>15} {'rel diff':>9}")
    f_sim, f_lin = [], []
    for ti in taus:
        r = run(ti, 0.02)
        f_sim.append(1000 / r["period"])
        f_lin.append(1000 * omega_lin(ti) / (2 * np.pi))
        print(f"{ti:6.1f} {f_sim[-1]:11.3f} {f_lin[-1]:15.3f} {(f_sim[-1] - f_lin[-1]) / f_lin[-1]:9.2e}")

    # --- figures -------------------------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    ax.loglog(eps_list, amps, "o", label="simulation")
    ee = np.geomspace(eps_list[0], eps_list[-1], 50)
    ax.loglog(ee, np.exp(icpt_all) * ee**slope_all, "-", label=f"fit, all $\\epsilon$: slope {slope_all:.3f}")
    ax.loglog(ee, np.exp(icpt_small) * ee**slope_small, "--",
              label=f"fit, $\\epsilon\\leq0.02$: slope {slope_small:.3f}")
    ax.set(xlabel=r"$\epsilon$  ($w_{EE}=(1+\epsilon)w_{EE}^{Hopf}$)", ylabel=r"amplitude $\max E-\min E$",
           title=r"(i) Amplitude vs $\epsilon$, $\tau_I=10$ ms")
    ax.legend(fontsize=8)

    ax = axes[1]
    tt = np.linspace(1.5, 22, 200)
    ax.plot(tt, 1000 * omega_lin(tt) / (2 * np.pi), "-", label=r"linear $\omega/2\pi$ from (b)")
    ax.plot(taus, f_sim, "o", label=r"simulation, $\epsilon=0.02$")
    ax.set(xlabel=r"$\tau_I$ (ms)", ylabel="frequency (Hz)", title=r"(ii) Frequency vs $\tau_I$")
    ax.legend(fontsize=8)

    ax = axes[2]
    r = run(10.0, 0.1)
    t = np.linspace(r["t_end"] - 60, r["t_end"], 3000)
    E, I = r["sol"].sol(t)
    t = t - t[0]
    ax.plot(t, E, label="E(t)")
    ax.plot(t, I, label="I(t)")
    ax.axhline(EBAR, color="gray", lw=0.6, ls=":")
    ax.set(xlabel="time (ms, shifted)", ylabel="activity",
           title=fr"(iii) $\tau_I=10$ ms, $\epsilon=0.1$: T={r['period']:.2f} ms, lag={r['lag']:.1f}$^\circ$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT, "problem1d_wilson_cowan.png")
    fig.savefig(path, dpi=150)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
