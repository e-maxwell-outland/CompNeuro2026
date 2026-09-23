"""Thompson sampling for two-patch foraging.

An animal chooses between two patches with unknown reward probabilities
p1 and p2. It keeps a Beta posterior over each, draws one sample from each
posterior, and forages in whichever patch drew higher. The chosen patch
pays 1 with its true probability, and its Beta is updated: alpha += reward,
beta += (1 - reward). Beta is conjugate to Bernoulli, so the posterior
after a successes and b failures from a uniform prior is Beta(1+a, 1+b).

The appeal is that exploration is automatic rather than tuned. Early on
the posteriors are broad and overlap, so choices are near random; as
evidence accumulates the better patch is sampled higher more often, and
exploration fades on its own. There is no epsilon to set, and no schedule
for annealing it, which is the practical contrast with the bandit module.

Replaces thompson_foraging.ipynb. Two changes: the run is averaged over
episodes, since a single 100-step run is too noisy to show that Thompson
beats chance, and cumulative regret is tracked, which is the quantity the
theory bounds.

Regression check: with p1 = 0.3, p2 = 0.6 the fraction of choices landing
on patch 2 should climb well above 0.5 within a hundred steps, and mean
regret per step should fall toward zero.
"""

import numpy as np

__all__ = ["thompson_run", "average_runs"]


def thompson_run(p=(0.3, 0.6), T=100, prior=(1.0, 1.0), rng=None):
    """One foraging episode. Returns cumulative reward, choices, regret."""
    rng = rng or np.random.default_rng()
    p = np.asarray(p, float)
    a = np.full(2, prior[0])
    b = np.full(2, prior[1])
    best = p.max()

    reward = np.zeros(T + 1)
    regret = np.zeros(T + 1)
    choice = np.zeros(T, dtype=int)
    for j in range(T):
        samp = rng.beta(a, b)
        c = int(np.argmax(samp))
        r = float(rng.random() < p[c])
        a[c] += r
        b[c] += 1 - r
        choice[j] = c
        reward[j + 1] = reward[j] + r
        regret[j + 1] = regret[j] + (best - p[c])
    return reward, choice, regret, (a, b)


def average_runs(p=(0.3, 0.6), T=100, runs=500, seed=0):
    """Mean reward, fraction choosing the better patch, and mean regret."""
    rng = np.random.default_rng(seed)
    R, C, G = [], [], []
    for _ in range(runs):
        reward, choice, regret, _ = thompson_run(p, T, rng=rng)
        R.append(reward); C.append(choice == int(np.argmax(p))); G.append(regret)
    return np.mean(R, axis=0), np.mean(C, axis=0), np.mean(G, axis=0)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import beta as beta_dist

    p = (0.3, 0.6)
    R, C, G = average_runs(p, T=200, runs=500)
    print(f"p = {p}, averaged over 500 runs of 200 steps:")
    print(f"  fraction on better patch: {C[:10].mean():.3f} (first 10 steps), "
          f"{C[-10:].mean():.3f} (last 10)")
    print(f"  cumulative regret {G[-1]:.2f}, mean per step {G[-1]/200:.4f}")

    _, _, _, (a, b) = thompson_run(p, T=200, rng=np.random.default_rng(1))
    x = np.linspace(0, 1, 500)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].plot(R, lw=2); ax[0].set_xlabel("step"); ax[0].set_ylabel("cumulative reward")
    w = 10
    ax[1].plot(np.convolve(C, np.ones(w) / w, "valid"), lw=2)
    ax[1].axhline(0.5, ls="--", color="k", lw=1)
    ax[1].set_xlabel("step"); ax[1].set_ylabel("fraction on better patch")
    for i, col in enumerate(["r", "b"]):
        ax[2].plot(x, beta_dist.pdf(x, a[i], b[i]), col, lw=2,
                   label=f"$p_{i+1}$ (true {p[i]})")
        ax[2].axvline(p[i], color=col, ls="--", lw=1)
    ax[2].set_xlabel("$p$"); ax[2].set_ylabel("posterior"); ax[2].legend()
    fig.tight_layout(); fig.savefig("thompson.pdf")
