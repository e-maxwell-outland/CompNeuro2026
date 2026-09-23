"""Neural field with linear adaptation, on a ring.

    u_t = -u - beta*v + int_{-pi}^{pi} w(x-y) H(u(y,t) - kappa) dy + I0*cos(x)
    tau*v_t = u - v

with cosine connectivity w(x-y) = cos(x-y) and a Heaviside firing rate of
threshold kappa. The cosine kernel is what makes this cheap: expanding
cos(x-y) = cos x cos y + sin x sin y turns the integral into two scalar
projections of the active region per step, rather than a full convolution.

Without adaptation (beta = 0) an initial bump settles into a stationary
one, the standard ring-attractor picture of working memory. Turn
adaptation on and add a weak input and the bump no longer sits still: the
slow variable v builds up behind it and pushes it away, so the bump drifts
or oscillates. That is the point of the model, and it is why the two
regimes below are worth running side by side.

Replaces adapt_nfield.ipynb, which had the same integration block twice
with only the parameters changed.

Regression check: with beta = 0 the bump centroid should stay put to
within a grid spacing. With beta = 0.2, tau = 12, I0 = 0.1 the centroid
should visibly move.
"""

import numpy as np

__all__ = ["simulate", "centroid"]


def simulate(beta=0.0, tau=10.0, I0=0.0, kappa=0.5, Nx=201, T=400.0, dt=0.05):
    """Integrate the field. Returns x, t, U (Nx by Nt), V."""
    L = np.pi
    x = np.linspace(-L, L, Nx)
    dx = 2 * L / (Nx - 1)
    Nt = int(round(T / dt)) + 1
    t = np.linspace(0, T, Nt)

    U = np.zeros((Nx, Nt))
    V = np.zeros((Nx, Nt))
    U[:, 0] = np.cos(x)
    if beta:
        V[:, 0] = np.cos(x + 0.5)

    sinx, cosx = np.sin(x), np.cos(x)
    for j in range(Nt - 1):
        H = (U[:, j] > kappa).astype(float)
        sinu = dx * H @ sinx
        cosu = dx * H @ cosx
        U[:, j + 1] = U[:, j] + dt * (-U[:, j] - beta * V[:, j]
                                      + sinu * sinx + (I0 + cosu) * cosx)
        V[:, j + 1] = V[:, j] + dt * (U[:, j] - V[:, j]) / tau
    return x, t, U, V


def centroid(x, U, kappa=0.5):
    """Circular centroid of the suprathreshold region at each time."""
    H = (U > kappa).astype(float)
    z = (np.exp(1j * x)[:, None] * H).sum(axis=0)
    return np.angle(np.where(np.abs(z) > 0, z, 1))


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    cases = [("no adaptation", dict(beta=0.0, tau=10.0, I0=0.0, T=400.0)),
             ("adaptation + input", dict(beta=0.2, tau=12.0, I0=0.10, T=1000.0))]
    for a, (lab, kw) in zip(ax, cases):
        x, t, U, V = simulate(**kw)
        c = centroid(x, U)
        drift = np.abs(c[-1] - c[0])
        print(f"{lab:20s}: centroid moved {drift:.4f} rad "
              f"({'stationary' if drift < 0.05 else 'moving'})")
        k = max(1, len(t) // 400)          # subsample for plotting only
        a.pcolormesh(x, t[::k], U[:, ::k].T, vmin=0.5, vmax=U.max(),
                     shading="auto", cmap="inferno")
        a.set_xlabel("$x$"); a.set_ylabel("time"); a.set_title(lab)
    fig.tight_layout(); fig.savefig("adapt_nfield.pdf")
