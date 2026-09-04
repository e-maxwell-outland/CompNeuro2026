"""Problem 4(c): LIF neuron driven by a synaptic transient.

Numerically integrates
    tau_m du/dt = -(u - u_rest) + R*I0*exp(-t/tau_s)
and checks the result against the closed-form expressions from 4(a)/4(b):
    u(t)    = R*I0*tau_s/(tau_s - tau_m) * (exp(-t/tau_s) - exp(-t/tau_m))
    t_peak  = tau_m*tau_s/(tau_s - tau_m) * log(tau_s/tau_m)
    u_max   = R*I0 * r**(-1/(r-1)),   r = tau_s/tau_m
    I0_crit = u_th / (R * r**(-1/(r-1)))
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

R = 1.0        # ohm
TAU_M = 10.0   # ms
TAU_S = 2.0    # ms
U_REST = 0.0   # mV
U_TH = 1.0     # mV


def _peak_ratio(r):
    """r**(-1/(r-1)), with the r=1 (tau_s = tau_m) removable singularity
    patched by its limit, e/... -> e^-1. Not hit at the given parameters,
    but the tau_s sweep in part (c) passes near tau_s = tau_m = 10 ms."""
    if abs(r - 1) < 1e-8:
        return np.e**-1
    return r ** (-1 / (r - 1))


def t_peak_analytic(tau_s, tau_m=TAU_M):
    return tau_m * tau_s / (tau_s - tau_m) * np.log(tau_s / tau_m)


def u_max_analytic(I0, tau_s=TAU_S, tau_m=TAU_M, R=R):
    r = tau_s / tau_m
    return R * I0 * _peak_ratio(r)


def I0_crit_analytic(tau_s=TAU_S, tau_m=TAU_M, R=R, u_th=U_TH):
    r = tau_s / tau_m
    return u_th / (R * _peak_ratio(r))


def simulate(I0, tmax, tau_s=TAU_S, tau_m=TAU_M, R=R, n=20000):
    """Integrate the LIF ODE directly (no closed form used here)."""

    def rhs(t, u):
        I = I0 * np.exp(-t / tau_s)
        return [(-(u[0] - U_REST) + R * I) / tau_m]

    t_eval = np.linspace(0, tmax, n)
    sol = solve_ivp(rhs, [0, tmax], [U_REST], t_eval=t_eval, rtol=1e-10, atol=1e-12)
    return sol.t, sol.y[0]


if __name__ == "__main__":
    tp = t_peak_analytic(TAU_S)
    Icrit = I0_crit_analytic(TAU_S)
    print(f"t_peak (analytic)  = {tp:.4g} ms")
    print(f"I0_crit (analytic) = {Icrit:.4g} mA")

    # Three amplitudes: below, at, and above I0_crit.
    amplitudes = {"below": 0.8 * Icrit, "at": Icrit, "above": 1.2 * Icrit}

    print(f"\n{'case':>6} {'I0 (mA)':>10} {'u_max analytic':>16} "
          f"{'u_max numeric':>15} {'t_peak numeric':>15}")

    fig, ax = plt.subplots()
    for label, I0 in amplitudes.items():
        t, u = simulate(I0, tmax=30)
        peak_idx = np.argmax(u)
        u_num, t_num = u[peak_idx], t[peak_idx]
        u_pred = u_max_analytic(I0)
        print(f"{label:>6} {I0:10.4g} {u_pred:16.4g} {u_num:15.4g} {t_num:15.4g}")
        ax.plot(t, u, label=f"{label} (I0={I0:.4g} mA)")

    ax.axhline(U_TH, color="k", linestyle="--", label="threshold")
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("u (mV)")
    ax.set_title("LIF response to a decaying synaptic current")
    ax.legend()
    fig.savefig("problem4c_traces.png", dpi=150)

    # ---- I0_crit vs tau_s on log-log axes ----
    tau_s_values = np.logspace(np.log10(0.5), np.log10(100), 200)
    Icrit_values = np.array([I0_crit_analytic(ts) for ts in tau_s_values])

    fig, ax = plt.subplots()
    ax.loglog(tau_s_values, Icrit_values)
    ax.set_xlabel(r"$\tau_s$ (ms)")
    ax.set_ylabel(r"$I_0^{\mathrm{crit}}$ (mA)")
    ax.set_title(r"Critical amplitude vs synaptic time constant ($\tau_m = 10$ ms)")
    fig.savefig("problem4c_Icrit_vs_taus.png", dpi=150)

    plt.show()
