"""Interspike interval statistics of a noisy integrate-and-fire neuron.

    du/dt = -alpha*u + Ibar + noise,   reset u -> 0 at u = 1

With subthreshold drive (Ibar below threshold) the neuron cannot fire
without noise at all, so every spike is noise-driven and the ISI
distribution is a first-passage density: sharply peaked with a long right
tail, nothing like the delta a deterministic model gives. Raising sigma
raises the firing rate even though the mean input has not changed, which
is the point of the model.

Two ways to collect ISIs, both here:
  - one long simulation, taking differences of spike times
  - many short first-passage runs, each stopped at threshold

The second gives a fixed sample size, so histograms are easier to compare
across parameters.

Uses lif_sim from lif_sim.py for the trace, rather than repeating the
integration loop as Noisy_IF.ipynb did twice more.

Fixed from the original: the first-passage loop wrote isi[j-1] instead of
isi[j], so entry 0 was never assigned and stayed exactly zero while entry
nsim-1 was written twice. One spurious zero interval in every sample
pulled the mean down and inflated the variance. The notebook also
demonstrated noise with rnd.rand() (uniform) where rnd.randn() (normal)
was meant.

Regression check: with Ibar = 0.95, sigma = 0.1 the two methods should
agree on the mean ISI to within sampling error. No ISI should be zero; if
one is, the off-by-one is back.
"""

import numpy as np
from lif_sim import lif_sim

__all__ = ["isi_from_trace", "isi_first_passage"]


def isi_from_trace(Ibar=0.95, sigma=0.1, taum=1.0, uth=1.0, T=5000.0,
                   dt=0.01, seed=0):
    """ISIs from one long run, as differences of spike times."""
    _, _, spikes, _ = lif_sim(Ibar, T=T, dt=dt, taum=taum, uth=uth,
                              sigma=sigma, seed=seed)
    return np.diff(spikes)


def isi_first_passage(Ibar=0.95, sigma=0.1, taum=1.0, uth=1.0, nsim=500,
                      dt=0.01, seed=0):
    """ISIs as first-passage times, one per run, fixed sample size."""
    rng = np.random.default_rng(seed)
    isi = np.zeros(nsim)
    for j in range(nsim):
        u, t = 0.0, 0.0
        while u < uth:
            u += dt * (Ibar - u) / taum + np.sqrt(2 * dt) * sigma * rng.standard_normal()
            t += dt
        isi[j] = t
    return isi


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = isi_from_trace()
    b = isi_first_passage()
    for lab, isi in [("long trace", a), ("first passage", b)]:
        print(f"{lab:14s}: n = {len(isi):5d}, mean ISI {isi.mean():8.3f}, "
              f"rate {1/isi.mean():.4f}, var {isi.var():.3f}, "
              f"min {isi.min():.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    t, u, spk, _ = lif_sim(0.4, T=20.0, sigma=0.6, seed=0)
    ax[0].plot(t, u, lw=1.2)
    ax[0].plot(spk, np.ones_like(spk), "r.", ms=8)
    ax[0].set_xlabel("$t$"); ax[0].set_ylabel("$u$")
    ax[0].set_title(r"noise-driven spiking, $\sigma=0.6$")

    ax[1].hist(b, 40, density=True, alpha=0.8)
    ax[1].set_xlabel("interspike interval"); ax[1].set_ylabel("pdf")
    ax[1].set_title("first-passage density")
    fig.tight_layout(); fig.savefig("noisy_if.pdf")
