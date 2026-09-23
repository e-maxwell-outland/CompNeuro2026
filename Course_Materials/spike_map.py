"""Spike map for a periodically forced integrate-and-fire neuron.

    u' = -u + I + beta*cos(2 pi t),   reset u -> 0 at u = 1

with the forcing period taken as 1 and the membrane time constant as 1.
If the neuron entrains 1:1, it spikes once per forcing cycle at a fixed
phase phi. Integrating from a reset at phase phi to threshold exactly one
period later gives the locking condition

    I + beta*(cos(2 pi phi) + 2 pi sin(2 pi phi))/(1 + 4 pi^2) = 1/(1 - e^-1).

CORRECTION: Spike_map.ipynb has a minus sign on the sine term. With the
minus the roots are 0.0875 and 0.4627; simulation locks at 0.9128, so the
formula as written does not describe the simulation it is compared to.
Deriving it, the integral int_0^1 e^tau sin(2 pi tau) dtau contributes
-2 pi (e-1)/(1+4 pi^2), and combining with the cosine term leaves a PLUS
on the sine. The corrected roots are 0.5373 and 0.9125; the second is the
stable one and matches simulation to four digits.

Regression check: I = 1.6, beta = 0.3 locks at phi = 0.9128 in simulation
and 0.9125 from the formula. If they disagree by more than 0.001, check
the sign on the sine term first.

Also fixed: the original seeded the spike list with the scalar 0, leaving
a phantom spike at t = 0 that shifted every phase computed from it.
"""

import numpy as np
from scipy.optimize import brentq

__all__ = ["forced_lif", "locking_condition", "locking_phases"]

_TWOPI = 2 * np.pi


def forced_lif(I=1.6, beta=0.3, T=200.0, dt=1e-4):
    """Integrate the forced LIF. Returns spike times and their phases."""
    nt = int(round(T / dt)) + 1
    t = np.linspace(0, T, nt)
    u = 0.0
    spikes = []
    for j in range(nt - 1):
        u += dt * (I + beta * np.cos(_TWOPI * t[j]) - u)
        if u > 1:
            u = 0.0
            spikes.append(t[j + 1])
    spikes = np.array(spikes)
    return spikes, spikes % 1


def locking_condition(phi, I=1.6, beta=0.3):
    """Residual of the 1:1 locking condition; zero at a locked phase."""
    return (I + beta * (np.cos(_TWOPI * phi) + _TWOPI * np.sin(_TWOPI * phi))
            / (1 + 4 * np.pi**2) - 1 / (1 - np.exp(-1)))


def locking_phases(I=1.6, beta=0.3, n=4001):
    """All roots of the locking condition on [0,1)."""
    xs = np.linspace(0, 1, n)
    v = locking_condition(xs, I, beta)
    return [brentq(locking_condition, xs[i], xs[i + 1], args=(I, beta))
            for i in range(n - 1) if v[i] * v[i + 1] < 0]


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spikes, phases = forced_lif()
    roots = locking_phases()
    print(f"simulation locks at phi = {phases[-1]:.4f}")
    print("theory roots:", [round(r, 4) for r in roots],
          "(the larger is the stable one)")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    t = np.linspace(0, 20, 20001)
    u = np.zeros_like(t)
    for j in range(len(t) - 1):
        u[j + 1] = u[j] + (t[1] - t[0]) * (1.6 + 0.3 * np.cos(_TWOPI * t[j]) - u[j])
        if u[j + 1] > 1:
            u[j + 1] = 0.0
    ax[0].plot(t, u, lw=1.2)
    for k in range(21):
        ax[0].axvline(k, color="r", lw=0.8, alpha=0.5)
    ax[0].set_xlabel("$t$"); ax[0].set_ylabel("$u$")
    ax[0].set_title("forcing cycles in red")

    ax[1].plot(np.arange(len(phases)), phases, "k.-", ms=5)
    for r in roots:
        ax[1].axhline(r, ls="--", lw=1, color="r")
    ax[1].set_xlabel("spike number"); ax[1].set_ylabel(r"phase $\phi$")
    ax[1].set_title("convergence to the stable locked phase")
    fig.tight_layout(); fig.savefig("spike_map.pdf")
