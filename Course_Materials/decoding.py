"""Signal detection and decoding from Poisson spike counts.

A neuron fires as a Poisson process with rate nu0 when a signal is absent
and nu1 > nu0 when it is present. A downstream readout counts spikes in a
window and calls "signal" when the count reaches a threshold k. Sweeping k
traces the ROC curve of detection probability against false alarm rate.

Pooling N identical neurons is equivalent to sampling one Poisson process
of rate N*nu0 or N*nu1. The means separate like N while the standard
deviations grow like sqrt(N), so discriminability improves like sqrt(N)
and the ROC curve bows further toward the corner.

Replaces Decoding.ipynb.

Fixed from the original: every loop ran to 20 (or 100) over arrays of
length 21 (or 101), so the last entry of each distribution and each
survival sum stayed zero. That truncation put a spurious point at the
origin of the ROC curve and slightly understated both PFA and PD near the
tail. Here the distributions come from scipy and the survival function is
exact.

Regression check: with nu0 = 2 and nu1 = 4 the error rate is minimized at
k = 3, between the two means. Area under the ROC should rise with N.
"""

import numpy as np
from scipy.stats import poisson

__all__ = ["roc", "error_rate", "auc"]


def roc(nu0=2.0, nu1=4.0, N=1, kmax=None):
    """Return thresholds k, false alarm rate, and detection rate.

    A trial is called "signal" when the pooled count is at least k, so
    PFA(k) = P(K >= k | nu0*N) and PD(k) = P(K >= k | nu1*N).
    """
    kmax = kmax or int(10 * N * nu1 + 20)
    k = np.arange(kmax + 1)
    PFA = poisson.sf(k - 1, N * nu0)      # sf(k-1) = P(K >= k)
    PD = poisson.sf(k - 1, N * nu1)
    return k, PFA, PD


def error_rate(PFA, PD):
    """Balanced error rate, assuming the two hypotheses are equally likely."""
    return (PFA + (1 - PD)) / 2


def auc(PFA, PD):
    """Area under the ROC curve, by the trapezoid rule."""
    order = np.argsort(PFA)
    return np.trapezoid(PD[order], PFA[order])


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nu0, nu1 = 2.0, 4.0
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))

    kk = np.arange(0, 21)
    ax[0].plot(kk, poisson.pmf(kk, nu0), lw=2, label=r"$\nu_0=2$ (absent)")
    ax[0].plot(kk, poisson.pmf(kk, nu1), lw=2, label=r"$\nu_1=4$ (present)")
    ax[0].set_xlabel("spike count $k$"); ax[0].set_ylabel("$P$"); ax[0].legend()

    ax[1].plot([0, 1], [0, 1], "k--", lw=1)
    for N in [1, 2, 5, 10]:
        k, PFA, PD = roc(nu0, nu1, N)
        ax[1].plot(PFA, PD, lw=2, label=f"$N={N}$ (AUC {auc(PFA,PD):.3f})")
        print(f"N = {N:3d}: AUC {auc(PFA, PD):.4f}, "
              f"min error rate {error_rate(PFA, PD).min():.4f} at k = "
              f"{k[np.argmin(error_rate(PFA, PD))]}")
    ax[1].set_xlabel("$P_{FA}$"); ax[1].set_ylabel("$P_D$"); ax[1].legend(fontsize=8)

    k, PFA, PD = roc(nu0, nu1, 1)
    ER = error_rate(PFA, PD)
    ax[2].plot(k[:21], ER[:21], lw=2)
    ax[2].plot(k[np.argmin(ER)], ER.min(), "ro")
    ax[2].set_xlabel("threshold $k$"); ax[2].set_ylabel("error rate")
    fig.tight_layout(); fig.savefig("decoding.pdf")
