"""Multi-armed bandits with epsilon-greedy action selection.

Ten arms. On each step the agent takes the arm with the highest running
value estimate, except with probability epsilon when it picks uniformly at
random. Estimates are updated by the sample-average rule
q_a <- q_a + (R - q_a)/n_a, which is exactly the running mean of the
rewards seen from arm a.

Two reward regimes, following Sutton and Barto section 2.3:
  bernoulli - each arm pays 1 with its own probability drawn uniform(0,1)
  gaussian  - each arm has a mean drawn N(0, rvar) and pays that mean plus
              N(0, svar) noise

The tradeoff to watch: epsilon = 0 is pure exploitation and often locks
onto whichever arm happened to pay first, so it plateaus below the others.
Larger epsilon finds the best arm more reliably but keeps paying a tax of
epsilon*(suboptimality) forever. Bernoulli arms are harder than Gaussian
ones here because a single binary sample carries so little information.

Replaces multiarmed_bandit.ipynb, which repeated the same 20-line loop six
times with only epsilon and the reward model changed. Averaging over runs
is added: a single run is dominated by which arm paid first, so the
epsilon comparison the notebook is making is not visible without it.

Regression check: with averaging over runs, the ordering of final average
reward should be epsilon = 0.1 > 0.01 > 0 for Bernoulli arms.
"""

import numpy as np

__all__ = ["run_bandit", "average_runs"]


def run_bandit(eps=0.1, T=2000, n_arms=10, kind="bernoulli",
               rvar=4.0, svar=1.0, rng=None):
    """One episode. Returns running average reward and fraction optimal."""
    rng = rng or np.random.default_rng()
    if kind == "bernoulli":
        truth = rng.uniform(0, 1, n_arms)
        draw = lambda a: float(rng.random() < truth[a])
    else:
        truth = rng.normal(0, rvar, n_arms)
        draw = lambda a: truth[a] + rng.normal(0, svar)
    kopt = int(np.argmax(truth))

    q = np.zeros(n_arms)
    n = np.zeros(n_arms)
    R = np.zeros(T + 1)
    O = np.zeros(T + 1)
    for j in range(T):
        a = rng.integers(n_arms) if rng.random() < eps else int(np.argmax(q))
        r = draw(a)
        n[a] += 1
        q[a] += (r - q[a]) / n[a]
        R[j + 1] = R[j] + (r - R[j]) / (j + 1)
        O[j + 1] = O[j] + ((a == kopt) - O[j]) / (j + 1)
    return R, O


def average_runs(eps, runs=200, seed=0, **kw):
    """Average reward and optimal-action curves over independent episodes."""
    rng = np.random.default_rng(seed)
    Rs, Os = [], []
    for _ in range(runs):
        R, O = run_bandit(eps=eps, rng=rng, **kw)
        Rs.append(R); Os.append(O)
    return np.mean(Rs, axis=0), np.mean(Os, axis=0)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    for row, kind in enumerate(["bernoulli", "gaussian"]):
        print(f"{kind} arms, averaged over 200 runs:")
        for eps in [0.0, 0.01, 0.1]:
            R, O = average_runs(eps, runs=200, kind=kind, T=1000)
            ax[row, 0].plot(R, lw=1.5, label=rf"$\epsilon={eps}$")
            ax[row, 1].plot(O, lw=1.5, label=rf"$\epsilon={eps}$")
            print(f"  eps = {eps:4}: final avg reward {R[-1]:.4f}, "
                  f"fraction optimal {O[-1]:.3f}")
        ax[row, 0].set_ylabel(f"{kind}\naverage reward")
        ax[row, 1].set_ylabel("fraction optimal")
        for a in ax[row]:
            a.set_xlabel("step"); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("bandit.pdf")
