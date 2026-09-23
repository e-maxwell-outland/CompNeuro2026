"""Problem 2c: NMDA current with magnesium block.

    I_N(V) = gbar B(V) (V - E_N),   B(V) = [1 + ([Mg]/3.57) exp(-V/V0)]^-1

dI_N/dV = gbar B [1 + (V - E_N)(1 - B)/V0], so with E_N = 0 the slope vanishes where
f(V) = 1 + V(1 - B)/V0 = 0. Found by bisection (no scipy in this environment).
gbar = 1 nS, so I_N is in pA when V is in mV.
"""

import numpy as np
import matplotlib.pyplot as plt

V0 = 16.13   # mV
E_N = 0.0    # mV
gbar = 1.0   # nS


def B(V, Mg):
    return 1.0 / (1.0 + (Mg / 3.57) * np.exp(-V / V0))


def I_N(V, Mg):
    return gbar * B(V, Mg) * (V - E_N)


def slope_factor(V, Mg):
    """dI_N/dV divided by gbar*B (> 0), so it has the same sign as the slope."""
    return 1.0 + (V - E_N) * (1.0 - B(V, Mg)) / V0


def bisect(f, lo, hi, tol=1e-8):
    flo = f(lo)
    if flo * f(hi) > 0:
        raise ValueError("root not bracketed")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if f(mid) * flo > 0:
            lo, flo = mid, f(mid)
        else:
            hi = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    Mg = 1.0
    V_star = bisect(lambda V: slope_factor(V, Mg), -90.0, 0.0)

    # Cross-check with a finite-difference derivative of I_N.
    V = np.linspace(-90, 0, 900001)
    dI = np.gradient(I_N(V, Mg), V)
    V_fd = V[np.where(np.diff(np.sign(dI)))[0][0]]

    print(f"[Mg2+] = 1 mM: dI_N/dV = 0 at V* = {V_star:.3f} mV (finite-difference check {V_fd:.3f} mV)")
    print(f"slope at -60 mV: {np.gradient(I_N(V, Mg), V)[np.argmin(abs(V + 60))]:.4f} nS (negative)")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in [0.0, 0.1, 1.0]:
        ax.plot(V, I_N(V, m), lw=2, label=rf"[Mg$^{{2+}}$] = {m:g} mM")
    ax.plot(V_star, I_N(V_star, Mg), "ko", ms=7,
            label=rf"$dI_N/dV=0$ at $V^*={V_star:.2f}$ mV (1 mM)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlim(-90, 0)
    ax.set_xlabel(r"membrane potential $V$ (mV)")
    ax.set_ylabel(r"$I_N$ (pA), $\bar g = 1$ nS")
    ax.set_title(r"NMDA current with Mg$^{2+}$ block, $E_N=0$ mV, $V_0=16.13$ mV")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("problem2c_nmda_IV.png", dpi=150)
