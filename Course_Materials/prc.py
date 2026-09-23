"""Phase response curve of the quadratic integrate-and-fire neuron.

For dv/dt = I + v^2 with I = 1, the natural period is Delta = pi and the
solution is v(t) = tan(t - pi/2) on one cycle. Labeling phase phi by time
since reset, the infinitesimal PRC for a voltage kick is

    Z(phi) = dphi/dv = 1/(1 + v^2) = sin^2(phi),

which vanishes at reset and at spike and peaks mid-cycle: a kick delivered
just after a spike does almost nothing, one delivered halfway through
advances the next spike the most.

This computes Z numerically by kicking at each phase and measuring how
much the next spike is advanced, then compares to sin^2.

Two fixes from Compute_PRC.ipynb. It allocated 102 entries but filled only
99, so the last three stayed zero and the plotted curve fell off a cliff
at the right edge that was an artifact, not physics. And the finite
threshold (100 instead of infinity) biases the measured period slightly;
that is kept but now stated, since it is the leading error here.

Regression check: max |Z_numeric - sin^2| should be a few percent at
kicksize = 0.05, and should shrink as kicksize does.
"""

import numpy as np

__all__ = ["qif_period", "prc_numeric", "prc_exact"]


def qif_period(I=1.0, dt=1e-4, thresh=100.0):
    """Numerical period, from reset at -thresh to crossing +thresh."""
    u, t = -thresh, 0.0
    while u <= thresh:
        u += dt * (I + u**2)
        t += dt
    return t


def prc_exact(phi, period):
    """Z(phi) = sin^2(pi*phi/period), the analytic infinitesimal PRC."""
    return np.sin(np.pi * np.asarray(phi) / period) ** 2


def prc_numeric(I=1.0, dt=1e-4, thresh=100.0, kicksize=0.05, nphase=100):
    """Kick at each phase, return (phases, Z) with Z = advance/kicksize."""
    period = qif_period(I, dt, thresh)
    phases = np.linspace(0, period, nphase)
    Z = np.zeros(nphase)
    for i, pk in enumerate(phases):
        u, t, kicked = -thresh, 0.0, False
        while u <= thresh:
            u += dt * (I + u**2)
            t += dt
            if not kicked and t >= pk:
                u += kicksize
                kicked = True
        Z[i] = (period - t) / kicksize
    return phases, Z, period


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phases, Z, period = prc_numeric()
    exact = prc_exact(phases, period)
    print(f"numerical period {period:.4f} (analytic pi = {np.pi:.4f})")
    print(f"max |Z_numeric - sin^2| = {np.abs(Z - exact).max():.4f}")

    fig, a = plt.subplots(figsize=(7, 4.2))
    a.plot(phases, exact, lw=3, label=r"$\sin^2\phi$ (analytic)")
    a.plot(phases, Z, "r.", ms=5, label="numerical")
    a.set_xlabel(r"phase $\phi$"); a.set_ylabel(r"$Z(\phi)$"); a.legend()
    fig.tight_layout(); fig.savefig("prc_qif.pdf")
