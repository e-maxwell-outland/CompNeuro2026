"""Kuramoto phase oscillators: a locked pair, and a population.

Pair:        theta_i' = omega_i + K sin(theta_j - theta_i)
Population:  theta_i' = omega_i + (K/N) sum_j sin(theta_j - theta_i)

Phases are kept on [0,1) rather than [0,2pi), so the sine arguments carry
an explicit 2*pi.

For the pair, the phase difference phi = theta_2 - theta_1 obeys
phi' = Delta - 2K sin(2 pi phi) with Delta = omega_2 - omega_1, so a fixed
phase difference exists exactly when 2K >= |Delta|. Identical oscillators
lock for arbitrarily weak coupling; a mismatched pair needs K >= |Delta|/2.
Regression check: omega = 1 and 1.4 gives Delta = 0.4, so locking requires
K >= 0.2. The demo runs K = 0.05 (drifting) and K = 0.2 (marginal).

For the population with frequencies drawn from a Cauchy density of width
gamma, the order parameter R = |mean of exp(2 pi i theta)| stays near zero
until K exceeds K_c = 2*gamma, then grows. numpy's standard_cauchy is
gamma = 1, so K_c = 2. Regression check: K = 1 leaves R small and noisy,
K = 10 drives R close to 1.

Consolidated from Kuramoto_model.ipynb, which had the pair loop three
times with different parameters, and wrote the population coupling with
two different meshgrid conventions. Those two are algebraically the same
sum written transposed; only one is kept here. Also fixed: the initial
condition hardcoded 100 instead of N, so changing N broke it.
"""

import numpy as np

__all__ = ["kuramoto_pair", "kuramoto_population"]


def kuramoto_pair(w1=1.0, w2=1.4, K=0.2, T=50.0, dt=1e-3, h0=(0.0, 0.45)):
    """Two coupled oscillators. Returns t, phase difference on [0,1)."""
    nt = int(round(T / dt)) + 1
    t = np.linspace(0, T, nt)
    h1 = np.zeros(nt); h2 = np.zeros(nt)
    h1[0], h2[0] = h0
    for j in range(nt - 1):
        h1[j + 1] = (h1[j] + dt * (w1 + K * np.sin(2 * np.pi * (h2[j] - h1[j])))) % 1
        h2[j + 1] = (h2[j] + dt * (w2 + K * np.sin(2 * np.pi * (h1[j] - h2[j])))) % 1
    return t, (h2 - h1) % 1


def kuramoto_population(N=100, K=1.0, T=20.0, dt=1e-3, gamma=1.0, seed=0):
    """Mean-field population with Cauchy frequencies. Returns t, R, psi."""
    rng = np.random.default_rng(seed)
    w = gamma * rng.standard_cauchy(N)
    nt = int(round(T / dt)) + 1
    t = np.linspace(0, T, nt)
    H = rng.random(N)
    R = np.zeros(nt); psi = np.zeros(nt)
    for j in range(nt - 1):
        # coupling[i] = sum_j sin(2 pi (H_j - H_i))
        coupling = np.sin(2 * np.pi * (H[None, :] - H[:, None])).sum(axis=1)
        H = (H + dt * (w + K * coupling / N)) % 1
        Z = np.exp(2j * np.pi * H).mean()
        R[j + 1] = np.abs(Z); psi[j + 1] = np.angle(Z)
    return t, R, psi


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for K, lab in [(0.05, "drifting"), (0.2, "locked")]:
        t, phi = kuramoto_pair(K=K)
        ax[0].plot(t, phi, lw=1.5, label=f"$K={K}$ ({lab})")
    ax[0].set_xlabel("$t$"); ax[0].set_ylabel(r"$\theta_2-\theta_1$")
    ax[0].set_title(r"pair, $\Delta=0.4$, locking needs $K\geq 0.2$")
    ax[0].legend(fontsize=9)

    for K in [1.0, 10.0]:
        t, R, _ = kuramoto_population(K=K)
        ax[1].plot(t, R, lw=1.5, label=f"$K={K:g}$")
        print(f"K = {K:5g}: final order parameter R = {R[-1]:.3f}")
    ax[1].set_xlabel("$t$"); ax[1].set_ylabel("order parameter $R$")
    ax[1].set_title(r"population, $K_c = 2\gamma = 2$")
    ax[1].set_ylim(0, 1); ax[1].legend(fontsize=9)
    fig.tight_layout(); fig.savefig("kuramoto_demo.pdf")
