"""FitzHugh-Nagumo model: time series, phase plane, and the Hopf boundary.

    u' = u - u^3/3 - w + I
    w' = eps*(u + b0 - gamma*w)

Replaces fn_mod.py and Fitzhugh-Nagumo.ipynb, which used two different
cubics and two different w equations. This is the convention used in the
course assignments; b0 = 0 and gamma = 1 recovers the assignment form
exactly, and b0 = 0.9 recovers the old notebook.

From the notebook notes: Fitzhugh (1961) and Nagumo et al (1962) reduced
Hodgkin-Huxley to two variables by noting that m is fast and that n and h
move together, leaving a fast cubic voltage variable u and a slow linear
recovery variable w. The cubic u-nullcline and the linear w-nullcline are
the whole story: where they cross sets the equilibrium, and how the slow
variable chases the fast one sets whether it spikes.

At gamma = 1 the equilibrium is unique, ubar = (3I)^(1/3), with
    trace J = 1 - ubar^2 - eps,   det J = eps*ubar^2 > 0,
so it is unstable exactly when ubar^2 < 1 - eps, i.e. |I| < I_c with
    I_c = (1 - eps)^(3/2)/3.
Regression check: eps = 3/4 gives I_c = 1/24 = 0.041667.

The Hopf is subcritical, so a large-amplitude cycle coexists with the
stable equilibrium for a range of I just above I_c. Perturb small to see
the linear prediction; perturb large and you can land on the cycle.
"""

import numpy as np
from scipy.integrate import solve_ivp

__all__ = ["fn_rhs", "fn_sim", "fn_hopf"]


def fn_rhs(t, y, I=0.0, eps=0.75, gamma=1.0, b0=0.0):
    u, w = y
    return [u - u**3 / 3 - w + I, eps * (u + b0 - gamma * w)]


def fn_sim(I=0.0, eps=0.75, gamma=1.0, b0=0.0, y0=(0.1, 0.0), T=200.0, dt=0.05):
    """Integrate the model. Returns t, u, w."""
    t = np.arange(0, T + 1e-9, dt)
    sol = solve_ivp(fn_rhs, [0, T], list(y0), args=(I, eps, gamma, b0),
                    t_eval=t, rtol=1e-9, atol=1e-11)
    return sol.t, sol.y[0], sol.y[1]


def fn_hopf(eps):
    """Critical current for the Hopf bifurcation at gamma = 1, b0 = 0."""
    return (1 - eps) ** 1.5 / 3 if eps < 1 else 0.0


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = 0.75
    Ic = fn_hopf(eps)
    print(f"eps = {eps}: I_c = {Ic:.6f} (exact 1/24 = {1/24:.6f})")

    # phase plane with nullclines and a trajectory, on either side of I_c
    ug = np.linspace(-2.5, 2.5, 400)
    U, W = np.meshgrid(np.linspace(-2.5, 2.5, 21), np.linspace(-1.5, 1.5, 21))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for a, I in zip(ax, [0.5 * Ic, 2.0 * Ic]):
        du, dw = fn_rhs(0, [U, W], I, eps)
        a.quiver(U, W, du, dw, color="0.7")
        a.plot(ug, ug - ug**3 / 3 + I, lw=2, label="$u$ nullcline")
        a.plot(ug, ug, lw=2, label="$w$ nullcline")
        _, u, w = fn_sim(I=I, eps=eps, T=400.0)
        a.plot(u, w, "c-", lw=1.2)
        ubar = np.cbrt(3 * I)
        a.plot(ubar, ubar, "ko", ms=6)
        a.set_xlim(-2.5, 2.5); a.set_ylim(-1.5, 1.5)
        a.set_xlabel("$u$"); a.set_ylabel("$w$")
        a.set_title(f"$I = {I:.4f}$ ({'below' if I < Ic else 'above'} $I_c$)")
        a.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig("fn_phaseplane.pdf")

    # time series
    fig, a = plt.subplots(figsize=(7, 3.6))
    t, u, w = fn_sim(I=0.5 * Ic, eps=eps, T=200.0)
    a.plot(t, u, lw=1.5, label="$u$")
    a.plot(t, w, lw=1.5, label="$w$")
    a.set_xlabel("$t$"); a.set_ylabel("$u$, $w$"); a.legend()
    fig.tight_layout()
    fig.savefig("fn_timeseries.pdf")

    for I in [0.5 * Ic, 2.0 * Ic]:
        _, u, _ = fn_sim(I=I, eps=eps, T=600.0)
        amp = u[-2000:].max() - u[-2000:].min()
        print(f"  I = {I:.5f}: amplitude {amp:.4f} -> "
              f"{'limit cycle' if amp > 1e-3 else 'fixed point'}")
