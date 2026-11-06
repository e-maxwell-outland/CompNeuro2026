"""Problem 3c: Hopfield network capacity.

N = 500 units, P = alpha N random +-1 patterns, alpha = 0.02, 0.06, ..., 0.62.
Hebb rule W = X X^T / N with W_ii = 0 (make_patterns/hebb_weights from Course_Materials/hopfield.py, which keeps
the diagonal, so it is zeroed here). Update s_i <- sgn(h_i), h = W s, sgn(0) = +1.

1. Unstable bits: fraction of (i, mu) with xi_i^mu h_i^mu < 0 when s = xi^mu. From part (b),
       xi_i h_i = (N-1)/N + C_i,  E[C_i] = 0,  Var C_i = (N-1)(P-1)/N^2 ~ alpha,
   so P(unstable) ~ Phi(-1/sqrt(alpha)); the finite-N version uses the exact mean and variance.
2. Recall: start from a stored pattern with 10% of bits (50) flipped, update asynchronously in a fresh random
   order each sweep until a full sweep changes nothing, and record m = xi^mu . s / N. 10 cues per alpha.

The dynamics run on the integer matrix J = N W = X X^T (zero diagonal), so h_i = 0 ties are exact.
Validation: along every asynchronous run each flip must change E = -s^T W s / 2 by dE = -(s_new - s_old) h_i,
strictly negative unless h_i = 0 (part a), the summed dE must equal E_final - E_start, and the final state must
satisfy sgn(W s) = s.
The empirical crosstalk mean/variance are checked against part (b) at one alpha.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Course_Materials"))
from hopfield import hebb_weights, make_patterns  # noqa: E402

N = 500
ALPHAS = np.round(np.arange(0.02, 0.63, 0.04), 2)
N_CUES, FLIP = 10, 0.10


def weights(X):
    """Integer J = N W with the diagonal zeroed (W = hebb_weights(X), entries are multiples of 1/N)."""
    J = np.rint(X.shape[0] * hebb_weights(X)).astype(np.int64)
    np.fill_diagonal(J, 0)
    return J


def energy(J, s):
    """N E(s), exact in integers."""
    return -(s @ J @ s) // 2


def recall_async(J, s, rng):
    """Asynchronous updates in random order until a full sweep changes nothing. Returns s, sweeps."""
    s = s.copy()
    h = J @ s
    E0, dE_sum, sweeps = energy(J, s), 0, 0
    while True:
        sweeps += 1
        changed = False
        for i in rng.permutation(len(s)):
            new = 1 if h[i] >= 0 else -1
            if new != s[i]:
                dE = -(new - s[i]) * h[i]
                assert dE < 0 or h[i] == 0, "a flip with h_i != 0 must strictly lower E"
                dE_sum += dE
                h += (new - s[i]) * J[:, i]
                s[i] = new
                changed = True
        if not changed:
            break
    assert energy(J, s) - E0 == dE_sum
    assert np.array_equal(np.where(J @ s >= 0, 1, -1), s), "final state is not a fixed point"
    return s, sweeps


def unstable_theory(alpha, N=N):
    P = np.round(alpha * N)
    return norm.cdf(-1 / np.sqrt(alpha)), norm.cdf(-((N - 1) / N) / np.sqrt((N - 1) * (P - 1) / N**2))


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    frac, frac_tie, m_mean, m_sd, m_all, sweeps_all = [], [], [], [], [], []
    print(f"N = {N}, cues flipped at {FLIP:.0%} ({int(FLIP*N)} bits), {N_CUES} cues per alpha")
    print(" alpha   P  | unstable sim  Phi(-1/sqrt a)  finite-N  | <m>     sd      min    | sweeps")
    for alpha in ALPHAS:
        P = int(round(alpha * N))
        X = make_patterns(N, P, rng).astype(np.int64)
        J = weights(X)
        XH = X * (J @ X) / N
        frac.append(np.mean(XH < 0))
        frac_tie.append(np.mean(XH == 0))
        if alpha == 0.14:
            C = XH - (N - 1) / N
            print(f"   [crosstalk check, alpha = {alpha}: mean C = {C.mean():+.4f} (theory 0), "
                  f"var C = {C.var():.4f} (theory (N-1)(P-1)/N^2 = {(N-1)*(P-1)/N**2:.4f})]")
        ms, sw = [], []
        for _ in range(N_CUES):
            mu = rng.integers(P)
            s = X[:, mu].copy()
            s[rng.choice(N, int(FLIP * N), replace=False)] *= -1
            s, k = recall_async(J, s, rng)
            ms.append(s @ X[:, mu] / N)
            sw.append(k)
        m_all.append(ms)
        m_mean.append(np.mean(ms)); m_sd.append(np.std(ms, ddof=1)); sweeps_all.append(np.mean(sw))
        th, thN = unstable_theory(alpha)
        print(f"  {alpha:.2f} {P:4d} |  {frac[-1]:.4e}    {th:.4e}     {thN:.4e} | "
              f"{m_mean[-1]:.4f}  {m_sd[-1]:.4f}  {min(ms):.4f} | {sweeps_all[-1]:.1f}")
    print(f"all {len(ALPHAS)*N_CUES} recall runs: E never increased (strict drop at every flip with h_i != 0), summed dE matched the E change, "
          f"final states are fixed points")
    print(f"ties xi h = 0 (counted stable under the xi h < 0 definition): max fraction {max(frac_tie):.2e}")
    a1 = 1 / norm.ppf(0.01) ** 2
    print(f"1% unstable load from part (b): alpha = {a1:.4f}")
    m_mean = np.array(m_mean)
    i_break = int(np.argmax(m_mean < 0.9))
    print(f"recall breakdown: <m> = {m_mean[i_break-1]:.4f} at alpha = {ALPHAS[i_break-1]:.2f}, "
          f"falls to {m_mean[i_break]:.4f} at alpha = {ALPHAS[i_break]:.2f}")

    # --- figure ---
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    af = np.linspace(0.01, 0.64, 300)
    th, thN = unstable_theory(af)
    ax[0].semilogy(ALPHAS, frac, "ko", ms=6, label="simulation (all $i,\\mu$)")
    ax[0].semilogy(af, th, "r-", lw=1.5, label=r"$\Phi(-1/\sqrt{\alpha})$")
    ax[0].semilogy(af, thN, "b--", lw=1,
                   label=r"finite $N$: $\Phi(-\frac{N-1}{N} / \sqrt{(N-1)(P-1)/N^2})$")
    ax[0].axhline(0.01, color="gray", lw=0.8, ls=":")
    ax[0].axvline(a1, color="gray", lw=0.8, ls=":", label=rf"1% unstable at $\alpha={a1:.3f}$")
    ax[0].set_xlabel(r"load $\alpha = P/N$")
    ax[0].set_ylabel(r"fraction of unstable bits, $\xi_i^\mu h_i^\mu < 0$")
    ax[0].set_ylim(1e-6, 0.3)
    ax[0].set_title(rf"bit stability of stored patterns, $N={N}$ (none unstable at $\alpha=0.02$)")
    ax[0].legend(fontsize=9)

    for a, ms in zip(ALPHAS, m_all):
        ax[1].plot(np.full(len(ms), a), ms, ".", color="gray", ms=4, alpha=0.6)
    m_min, m_max = np.array([min(ms) for ms in m_all]), np.array([max(ms) for ms in m_all])
    ax[1].errorbar(ALPHAS, m_mean, yerr=[m_mean - m_min, m_max - m_mean], fmt="ko-", ms=5, capsize=3,
                   label=r"$\langle m\rangle$, bars: min$-$max over cues")
    ax[1].plot([], [], ".", color="gray", label="individual cues")
    ax[1].axhline(1 - 2 * FLIP, color="b", ls=":", lw=1, label=rf"initial overlap $m_0={1-2*FLIP:.1f}$")
    ax[1].set_xlabel(r"load $\alpha = P/N$")
    ax[1].set_ylabel(r"final overlap $m = \xi^\mu\cdot s/N$")
    ax[1].set_title(rf"async recall from 10% flipped cue, $N={N}$, {N_CUES} cues per $\alpha$")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("problem3c_hopfield.png", dpi=150)
