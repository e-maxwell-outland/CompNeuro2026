"""Problem 2c: Phase response curves by direct perturbation.

At phase phi (time phi*Delta after the phase-zero event) kick the voltage by delta, record the time T'
of a later marker event, and compute
    R(phi) ~ (T_ref - T') / (Delta delta).

T_ref is the same run with delta = 0, so integration error cancels instead of entering as T_ref = n*Delta.
Phases sit at grid midpoints (k + 1/2)/N so a kick never lands exactly on the marker event.

LIF  dv/dt = -v/tau + I, tau = 1, v_r = 0, v_th = 1, I = 1.5. Marker: the next threshold crossing.
     From part a: Delta = tau log[(I tau - v_r)/(I tau - v_th)] = log 3,
                  R(phi) = tau e^{phi Delta/tau} / (Delta (I tau - v_r)).

FHN  dv/dt = v - v^3/3 - w + 0.5,  dw/dt = 0.08 (v + 0.7 - 0.8 w)  (Course_Materials/fn_sim.py form,
     with I = 0.5, eps = 0.08, b0 = 0.7, gamma = 0.8). No reset: phase zero is the upward crossing of
     v = 0, and the marker is the n-th upward crossing after the kick, by which time the perturbation
     has relaxed back onto the limit cycle.

The same measure_prc routine does both, so agreement with the LIF formula validates it before it is
trusted on FHN, where there is no closed form.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

TOL = dict(rtol=1e-10, atol=1e-12)


def lif_rhs(t, y, I=1.5, tau=1.0):
    return [-y[0] / tau + I]


def lif_threshold(t, y):
    return y[0] - 1.0
lif_threshold.direction = 1


def fhn_rhs(t, y):
    v, w = y
    return [v - v**3 / 3 - w + 0.5, 0.08 * (v + 0.7 - 0.8 * w)]


def fhn_upcross(t, y):
    return y[0]
fhn_upcross.direction = 1


def nth_event(rhs, y, t0, event, n, span):
    """Time of the n-th event after t0, integrating at most `span` time units."""
    sol = solve_ivp(rhs, [t0, t0 + span], y, events=event, **TOL)
    return sol.t_events[0][n - 1]


def measure_prc(rhs, y0, period, event, n, delta, phis, threshold=None):
    """R(phi) by kicking component 0 of the state by delta at time phi*period after y0.

    threshold: for a reset model, a kick that carries v past it fires immediately (T' = kick time).
    """
    orbit = solve_ivp(rhs, [0, period], y0, dense_output=True, **TOL).sol
    span = (n + 1) * period
    R = np.empty(len(phis))
    for i, phi in enumerate(phis):
        tk = phi * period
        y = orbit(tk)
        T_ref = nth_event(rhs, y, tk, event, n, span)
        yk = y.copy()
        yk[0] += delta
        if threshold is not None and yk[0] >= threshold:
            T_kick = tk
        else:
            T_kick = nth_event(rhs, yk, tk, event, n, span)
        R[i] = (T_ref - T_kick) / (period * delta)
    return R


def fhn_cycle(t_settle=500.0):
    """State at an upward v = 0 crossing on the limit cycle, and the period."""
    s = solve_ivp(fhn_rhs, [0, t_settle], [0.0, 0.0], **TOL)
    s = solve_ivp(fhn_rhs, [0, 300.0], s.y[:, -1], events=fhn_upcross, **TOL)
    te = s.t_events[0]
    return s.y_events[0][-1], np.diff(te).mean()


def negative_intervals(phis, R):
    """Phase intervals where R < 0, located to grid resolution."""
    neg = R < 0
    out, i = [], 0
    while i < len(R):
        if neg[i]:
            j = i
            while j + 1 < len(R) and neg[j + 1]:
                j += 1
            out.append((phis[i], phis[j]))
            i = j + 1
        else:
            i += 1
    return out


if __name__ == "__main__":
    N = 200
    phis = (np.arange(N) + 0.5) / N
    delta = 1e-3

    # --- LIF ---
    I, tau, vr = 1.5, 1.0, 0.0
    D_lif = nth_event(lif_rhs, [vr], 0.0, lif_threshold, 1, 10.0)
    R_lif = measure_prc(lif_rhs, [vr], D_lif, lif_threshold, 1, delta, phis, threshold=1.0)
    R_lif_th = tau * np.exp(phis * D_lif / tau) / (D_lif * (I * tau - vr))
    R_lif_half = measure_prc(lif_rhs, [vr], D_lif, lif_threshold, 1, delta / 2, phis, threshold=1.0)
    e1, e2 = np.abs(R_lif - R_lif_th).max(), np.abs(R_lif_half - R_lif_th).max()
    print(f"LIF (tau={tau}, v_r={vr}, v_th=1, I={I}):")
    print(f"  period {D_lif:.5f} (analytic log 3 = {np.log(3):.5f})")
    print(f"  max |R_num - R_formula| = {e1:.2e} at delta={delta:g}, {e2:.2e} at delta={delta/2:g} "
          f"(ratio {e1/e2:.2f}, O(delta) -> 2)")
    print(f"  R range [{R_lif.min():.4f}, {R_lif.max():.4f}] "
          f"(formula R(0) = {1/(D_lif*I):.4f}, R(1) = {np.exp(D_lif)/(D_lif*I):.4f}); "
          f"R < 0 nowhere -> Type I")

    # --- FHN ---
    y0, D_fhn = fhn_cycle()
    n = 4
    R_fhn = measure_prc(fhn_rhs, y0, D_fhn, fhn_upcross, n, delta, phis)
    R_d = measure_prc(fhn_rhs, y0, D_fhn, fhn_upcross, n, delta / 2, phis)
    R_n = measure_prc(fhn_rhs, y0, D_fhn, fhn_upcross, n - 1, delta, phis)
    imax, imin = R_fhn.argmax(), R_fhn.argmin()
    print(f"\nFHN (phase 0 at upward v = 0 crossing, marker = crossing {n} after the kick):")
    print(f"  period {D_fhn:.4f}, phase-zero state (v, w) = ({y0[0]:.4f}, {y0[1]:.4f})")
    print(f"  max R = {R_fhn[imax]:.4f} at phi = {phis[imax]:.4f}")
    print(f"  min R = {R_fhn[imin]:.4f} at phi = {phis[imin]:.4f}")
    for a, b in negative_intervals(phis, R_fhn):
        print(f"  R < 0 for phi in [{a:.4f}, {b:.4f}] (grid step {1/N:.4f}) -> Type II")
    print(f"  consistency: max |R(delta) - R(delta/2)| = {np.abs(R_fhn - R_d).max():.2e}, "
          f"max |R(n={n}) - R(n={n-1})| = {np.abs(R_fhn - R_n).max():.2e}")

    # --- figure ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    pc = np.linspace(0, 1, 400)
    ax[0].plot(pc, tau * np.exp(pc * D_lif / tau) / (D_lif * (I * tau - vr)), "r", lw=2,
               label=r"formula $\tau e^{\phi\Delta/\tau}/[\Delta(I\tau-v_r)]$")
    ax[0].plot(phis[::4], R_lif[::4], "k.", ms=6, label=rf"numerical, $\delta={delta:g}$")
    ax[0].set_xlabel(r"phase of kick $\phi$ (fraction of period)")
    ax[0].set_ylabel(r"$R(\phi)$ (phase advance per unit voltage kick)")
    ax[0].set_title(rf"LIF: $\tau=1$, $v_r=0$, $v_{{th}}=1$, $I=1.5$, $\Delta=\log 3={D_lif:.4f}$")
    ax[0].legend(fontsize=9)

    ax[1].plot(phis, R_fhn, "k.-", ms=3, lw=1, label=rf"numerical, $\delta={delta:g}$, {n} cycles later")
    ax[1].axhline(0, color="gray", lw=0.8)
    ax[1].fill_between(phis, R_fhn, 0, where=R_fhn < 0, color="b", alpha=0.3, label=r"$R<0$ (delay)")
    ax[1].set_xlabel(r"phase of kick $\phi$ (0 = upward crossing of $v=0$)")
    ax[1].set_ylabel(r"$R(\phi)$ (phase advance per unit voltage kick)")
    ax[1].set_title(rf"FitzHugh-Nagumo, $I=0.5$, $\epsilon=0.08$, period $\Delta={D_fhn:.3f}$")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("problem2c_prc.png", dpi=150)
