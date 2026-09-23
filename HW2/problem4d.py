"""Problem 4d: Direct simulation of Poisson shot noise.

    tau dV/dt = -V + tau sum_k w delta(t - t_k),   t_k Poisson of rate nu

<V> = 10 mV and tau = 20 ms are held fixed while nu*tau = 2, 20, 200, so w = <V>/(nu tau).
Theory: Var = nu w^2 tau / 2,  skewness = 2 sqrt(2) / (3 sqrt(nu tau)).

V is sampled exactly on a grid of spacing dt: each event contributes w exp(-(t_grid - t_k)/tau)
to the next grid point, and between grid points V decays by exp(-dt/tau). Nothing is discretized.
Samples are correlated over ~tau, so a long run is needed to resolve the small skewness at nu tau = 200.
"""

import numpy as np
import matplotlib.pyplot as plt


def shot_noise(nu, w, tau, T, dt, rng):
    """V on the grid t_j = j*dt, j = 0..n, started from V = 0."""
    n = int(round(T / dt))
    # Poisson process on [0, T]: Poisson(nu T) events placed uniformly.
    nev = rng.poisson(nu * T)
    t_ev = np.sort(rng.uniform(0.0, T, nev))
    # Event in (t_j, t_{j+1}] contributes to grid point j+1.
    j_next = np.minimum(np.ceil(t_ev / dt).astype(int), n)
    kick = w * np.exp(-(j_next * dt - t_ev) / tau)
    c = np.bincount(j_next, weights=kick, minlength=n + 1)

    decay = np.exp(-dt / tau)
    V = np.empty(n + 1)
    V[0] = 0.0
    v = 0.0
    for j in range(1, n + 1):
        v = v * decay + c[j]
        V[j] = v
    return V


def skewness(x):
    d = x - x.mean()
    return np.mean(d**3) / np.mean(d**2) ** 1.5


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    Vmean = 10.0     # mV
    tau = 0.020      # s
    dt = 0.0002      # s (sampling grid; exact, so this only sets the sample spacing)
    T = 1000.0       # s
    burn = 10 * tau  # s

    rows, keep = [], {}
    for nt in [2, 20, 200]:
        nu = nt / tau
        w = Vmean / nt
        V = shot_noise(nu, w, tau, T, dt, rng)[int(burn / dt):]
        th = (nu * w * tau, nu * w**2 * tau / 2, 2 * np.sqrt(2) / (3 * np.sqrt(nt)))
        sim = (V.mean(), V.var(), skewness(V))
        rows.append((nt, nu, w, th, sim))
        if nt in (2, 200):
            keep[nt] = (V, th)

    print(f"tau = {tau*1e3:.0f} ms, <V> = {Vmean:.0f} mV, T = {T:.0f} s, grid dt = {dt*1e3:.1f} ms")
    print(f"{'nu*tau':>6} {'nu (Hz)':>8} {'w (mV)':>7} | {'mean th':>8} {'mean sim':>8} | "
          f"{'var th':>8} {'var sim':>8} | {'skew th':>8} {'skew sim':>8}")
    for nt, nu, w, th, sim in rows:
        print(f"{nt:>6} {nu:>8.0f} {w:>7.3f} | {th[0]:>8.3f} {sim[0]:>8.3f} | "
              f"{th[1]:>8.4f} {sim[1]:>8.4f} | {th[2]:>8.4f} {sim[2]:>8.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, nt in zip(ax, [2, 200]):
        V, th = keep[nt]
        sd = np.sqrt(th[1])
        a.hist(V, bins=150, density=True, alpha=0.6, label="simulation")
        x = np.linspace(V.min(), V.max(), 500)
        a.plot(x, np.exp(-(x - th[0])**2 / (2 * th[1])) / np.sqrt(2 * np.pi * th[1]), "r", lw=1,
               label=rf"Gaussian $N(\langle V\rangle, \nu w^2\tau/2)$, sd = {sd:.3g} mV")
        a.set_xlabel(r"$V$ (mV, from rest)")
        a.set_ylabel("probability density (1/mV)")
        a.set_title(rf"$\nu\tau={nt}$: $\nu={nt/tau:.0f}$ Hz, $w={Vmean/nt:g}$ mV, $\tau=20$ ms")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("problem4d_histograms.png", dpi=150)
