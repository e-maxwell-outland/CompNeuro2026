"""Problem 5c: Euler-Maruyama simulation of the perfect (non-leaky) noisy integrator.

    du/dt = mu + sqrt(2 sigma^2) xi(t),   spike and reset u -> 0 when u >= theta

Same noise convention as Course_Materials/lif_sim.py (increment sqrt(2 dt) sigma randn), but no
leak, and an ensemble of independent neurons is stepped together instead of one long trace.

Predictions from parts a-b: nu = mu/theta, P(u < 0) = (sigma^2/(mu theta)) (1 - exp(-mu theta/sigma^2)),
and the stationary density p(u) plotted below.

Every neuron starts at the reset u = 0, so the region below reset (length scale sigma^2/mu) starts
empty and fills in slowly by diffusion. The first t_burn time units are discarded for that reason.
"""

import numpy as np
import matplotlib.pyplot as plt


def p_theory(u, mu, s2, theta):
    nu = mu / theta
    return np.where(u >= 0,
                    nu / mu * (1 - np.exp(mu * (u - theta) / s2)),
                    nu / mu * (1 - np.exp(-mu * theta / s2)) * np.exp(mu * u / s2))


def simulate(mu, s2, theta, N, dt, t_burn, t_run, rng, sample_every=100):
    u = np.zeros(N)
    n_burn, n_run = int(round(t_burn / dt)), int(round(t_run / dt))
    noise = np.sqrt(2 * s2 * dt)
    spikes, below, count, samples = 0, 0, 0, []
    for j in range(n_burn + n_run):
        u += mu * dt + noise * rng.standard_normal(N)
        fired = u >= theta
        u[fired] = 0.0
        if j >= n_burn:
            spikes += fired.sum()
            below += (u < 0).sum()
            count += N
            if j % sample_every == 0:
                samples.append(u.copy())
    rate = spikes / (N * t_run)
    frac_below = below / count
    return rate, frac_below, np.concatenate(samples)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    mu, theta = 1.0, 1.0
    N, dt, t_burn, t_run = 500, 1e-4, 20.0, 50.0

    print(f"mu = theta = {mu:g}, N = {N}, dt = {dt:g}, burn-in = {t_burn:g}, run = {t_run:g}")
    print(f"{'sigma^2':>7} | {'rate th':>7} {'rate sim':>8} | {'P(u<0) th':>9} {'P(u<0) sim':>10}")
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for a, s2 in zip(ax, [0.25, 1.0, 4.0]):
        rate, frac, us = simulate(mu, s2, theta, N, dt, t_burn, t_run, rng)
        frac_th = s2 / (mu * theta) * (1 - np.exp(-mu * theta / s2))
        print(f"{s2:>7g} | {mu/theta:>7.4f} {rate:>8.4f} | {frac_th:>9.4f} {frac:>10.4f}")

        a.hist(us, bins=200, density=True, alpha=0.6, label="simulation")
        x = np.linspace(us.min(), theta, 1000)
        a.plot(x, p_theory(x, mu, s2, theta), "r", lw=1, label=r"theory $p(u)$")
        a.axvline(0, color="k", ls="--", lw=1, label="reset $u=0$")
        a.set_xlabel(r"$u$ (units of $\theta$)")
        a.set_ylabel("probability density")
        a.set_title(rf"$\sigma^2={s2:g}$: rate {rate:.3f} (th 1), $P(u<0)$ {frac:.3f} (th {frac_th:.3f})",
                    fontsize=10)
        a.legend(fontsize=8)
    fig.suptitle(r"Perfect integrator with noise, $\mu=\theta=1$, Euler-Maruyama $dt=10^{-4}$")
    fig.tight_layout()
    fig.savefig("problem5c_density.png", dpi=150)
