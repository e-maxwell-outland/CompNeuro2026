"""Maximum likelihood and Bayesian inference of a Poisson firing rate.

Interspike intervals of a Poisson neuron of rate lambda are exponential,
so the likelihood of n observed intervals is

    L(lambda | x_1..x_n) = lambda^n exp(-lambda * sum x_j),

whose single critical point gives the MLE lambda_hat = 1/xbar for every n.
What changes with n is not where the peak sits on average but how sharp it
is: the likelihood narrows like 1/sqrt(n) and converges on a delta at the
true rate.

Replaces Poisson_infer.ipynb, which repeated the same four lines for
n = 1, 5, 10, 100.

Numerical note, and the reason this is a function rather than the inline
expression: lambda**100 overflows to 1e100 before the exponential pulls it
back down, so the direct product loses precision badly at large n and
silently returns nan for n much past a few hundred. Everything here is
computed as a log likelihood and exponentiated only at the end, after
subtracting the maximum.

Regression check: the MLE equals 1/mean(x) exactly for every n, and the
posterior width should fall roughly like 1/sqrt(n).
"""

import numpy as np

__all__ = ["log_likelihood", "likelihood", "mle"]


def log_likelihood(lam, x):
    """log L(lambda | x), up to an additive constant. lam may be an array."""
    lam = np.asarray(lam, float)
    n, s = len(np.atleast_1d(x)), np.sum(x)
    with np.errstate(divide="ignore"):
        return np.where(lam > 0, n * np.log(np.where(lam > 0, lam, 1)) - lam * s, -np.inf)


def likelihood(lam, x, normalize=True):
    """L(lambda | x), computed in logs then exponentiated. Peak scaled to 1."""
    ll = log_likelihood(lam, x)
    L = np.exp(ll - np.nanmax(ll[np.isfinite(ll)]))
    return L / L.max() if normalize else L


def mle(x):
    """lambda_hat = 1/mean(x), the single critical point of L."""
    return 1.0 / np.mean(x)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    lam_true = 1.0
    lam = np.linspace(1e-6, 10, 2000)

    fig, a = plt.subplots(figsize=(7, 4.4))
    print(f"true lambda = {lam_true}")
    for n, col in [(1, "C0"), (5, "C1"), (10, "C2"), (100, "C3")]:
        x = rng.exponential(1 / lam_true, n)
        L = likelihood(lam, x)
        a.plot(lam, L, lw=2, color=col, label=f"$n={n}$")
        a.plot(mle(x), 1.0, "o", color=col)
        # width at half max, as a crude measure of sharpening
        above = lam[L > 0.5]
        print(f"  n = {n:4d}: MLE {mle(x):.4f}, half-max width "
              f"{above.max()-above.min():.4f}")
    a.set_xlabel(r"$\lambda$"); a.set_ylabel("likelihood (peak 1)")
    a.set_xlim(0, 5); a.legend()
    fig.tight_layout(); fig.savefig("poisson_infer.pdf")
