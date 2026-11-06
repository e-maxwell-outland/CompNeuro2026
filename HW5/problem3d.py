"""Problem 3d: Stability of Amari bumps for the Mexican-hat kernel w(x) = 2 e^{-|x|} - e^{-|x|/2}.

    du/dt = -u + int w(x-y) f(u(y)) dy,   f(u) = 1/(1 + e^{-100(u - kappa)})   (steep stand-in for H(u - kappa)),
on x in [-20, 20], dx = 0.02 (2001 points), explicit Euler with dt = 0.02. The convolution is a linear (not circular)
one: the kernel is sampled at all offsets -(N-1)dx..(N-1)dx, both arrays are zero-padded to a power of two
>= 3N - 2 and multiplied in Fourier space, and the result is scaled by dx (rectangle rule).

Initial data u(x, 0) = W(x + a0) - W(x - a0), with W(x) = sign(x)(2 e^{-|x|/2} - 2 e^{-|x|}) the odd antiderivative of
w from part (a); for a0 = a_+- this is exactly the stationary bump.
Half-width a(t): the left and right crossings of u = kappa (linear interpolation between grid points), averaged;
a = 0 once u < kappa everywhere (the bump has died).

From parts (a), (b), with S = sqrt(1 - 2 kappa):
    e^{-a_-} = (1 + S)/2 (narrow),  e^{-a_+} = (1 - S)/2 (wide),
    lambda_e(a_-) = S(1+S)/(1 - S(1+S)/2) > 0,   lambda_e(a_+) = -S(1-S)/(1 + S(1-S)/2) < 0.

Validation:
  * the FFT convolution must equal a direct np.convolve to round-off, and the convolution of the indicator of
    |x| < a must reproduce W(x+a) - W(x-a) to O(dx);
  * started exactly at a_+ the bump must stay put: |a(inf) - a_+| is the size of the discretisation/sigmoid shift;
  * the relaxation rate is fitted to log|a(t) - a(inf)| for 1e-7 <= |a - a(inf)| <= 1e-2, with a(inf) taken from the
    end of the run rather than the analytic a_+, because the small equilibrium shift a(inf) - a_+ ~ 1e-4 would
    otherwise bend the log plot at late times.

The fitted rates come out ~3% faster than lambda_e(a_+) at kappa = 0.3, 0.45. That offset is from the sigmoid, not the
grid or time step: at kappa = 0.3, halving dx twice leaves the rate at -0.216, while steepness 50/100/200/400 (with dx
fine enough to resolve the edge) gives -0.221/-0.216/-0.213/-0.212, closing on -0.208 as f -> H. Steepness > ~200 at
dx = 0.02 is not usable: the edge becomes narrower than a grid cell and pins.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

L, DX, DT = 20.0, 0.02, 0.02
X = np.round(np.arange(-L, L + DX / 2, DX), 10)
N = len(X)
OUT = os.path.dirname(os.path.abspath(__file__))


def w(x):
    return 2 * np.exp(-np.abs(x)) - np.exp(-np.abs(x) / 2)


def W(x):
    return np.sign(x) * (2 * np.exp(-np.abs(x) / 2) - 2 * np.exp(-np.abs(x)))


# linear convolution by zero-padded FFT
OFFS = DX * np.arange(-(N - 1), N)
M = 1 << int(np.ceil(np.log2(3 * N - 2)))
KHAT = np.fft.rfft(w(OFFS), M)


def conv(F):
    full = np.fft.irfft(np.fft.rfft(F, M) * KHAT, M)[: 3 * N - 2]
    return DX * full[N - 1: 2 * N - 1]


def widths(kappa):
    S = np.sqrt(1 - 2 * kappa)
    a_minus, a_plus = -np.log((1 + S) / 2), -np.log((1 - S) / 2)
    lam_minus = S * (1 + S) / (1 - S * (1 + S) / 2)
    lam_plus = -S * (1 - S) / (1 + S * (1 - S) / 2)
    return a_minus, a_plus, lam_minus, lam_plus


def half_width(u, kappa):
    above = np.nonzero(u > kappa)[0]
    if len(above) == 0:
        return 0.0
    i, j = above[0], above[-1]
    xl = X[i - 1] + (kappa - u[i - 1]) / (u[i] - u[i - 1]) * DX
    xr = X[j] + (u[j] - kappa) / (u[j] - u[j + 1]) * DX
    return 0.5 * (xr - xl)


def simulate(kappa, a0, T, every=5):
    u = W(X + a0) - W(X - a0)
    ts, As = [0.0], [half_width(u, kappa)]
    nsteps = int(round(T / DT))
    for n in range(1, nsteps + 1):
        u = u + DT * (-u + conv(expit(100 * (u - kappa))))
        if n % every == 0:
            ts.append(n * DT)
            As.append(half_width(u, kappa))
            if As[-1] == 0.0 and np.max(u) < kappa - 0.05:
                break
    return np.array(ts), np.array(As), u


def fit_rate(t, d, lo, hi):
    """Slope of log d against t over the stretch lo <= d <= hi (after d first drops below hi)."""
    start = np.argmax(d <= hi)
    sel = np.arange(len(d)) >= start
    sel &= (d >= lo) & (d <= hi)
    k, c = np.polyfit(t[sel], np.log(d[sel]), 1)
    return k, c, sel


def main():
    # --- validation of the convolution ---------------------------------------------------------------------------
    rng = np.random.default_rng(0)
    F = rng.random(N)
    err_direct = np.max(np.abs(conv(F) - DX * np.convolve(F, w(OFFS))[N - 1: 2 * N - 1]))
    a_test = 1.5
    err_W = np.max(np.abs(conv((np.abs(X) < a_test).astype(float)) - (W(X + a_test) - W(X - a_test))))
    print(f"conv check: FFT vs direct max err = {err_direct:.1e}; indicator vs W(x+a)-W(x-a) max err = {err_W:.2e} "
          f"(dx = {DX})\n")
    assert err_direct < 1e-10 and err_W < DX

    # --- kappa = 0.3: which perturbed bumps return, run away, die -----------------------------------------------
    kappa = 0.3
    am, ap, lm, lp = widths(kappa)
    print(f"kappa = {kappa}: a_- = {am:.4f}, a_+ = {ap:.4f}, U(0)_- = {2 * W(am):.4f}, U(0)_+ = {2 * W(ap):.4f}, "
          f"lambda_e(a_-) = {lm:.4f}, lambda_e(a_+) = {lp:.4f}")
    cases = [(am - 0.02, r"$a_- - 0.02$"), (am + 0.02, r"$a_- + 0.02$"),
             (ap - 0.05, r"$a_+ - 0.05$"), (ap + 0.05, r"$a_+ + 0.05$")]
    runs = []
    for a0, lab in cases:
        t, A, u = simulate(kappa, a0, 120)
        runs.append((t, A, lab))
        fate = "died" if A[-1] == 0 else f"final a = {A[-1]:.4f}, max u = {u.max():.4f}"
        print(f"   a0 = {a0:.4f}: a(0) = {A[0]:.4f} -> {fate} (t = {t[-1]:.1f})")
    print()

    # --- relaxation rates of the wide bump ------------------------------------------------------------------------
    print(f"{'kappa':>6} {'a_+':>7} {'a(inf)':>8} {'a(inf)-a_+':>11} {'fitted rate':>12} "
          f"{'lambda_e(a_+)':>14} {'rel err':>8}")
    rel = []
    for kap in [0.1, 0.3, 0.45]:
        am_k, ap_k, _, lp_k = widths(kap)
        T = 25 / abs(lp_k)
        _, A_eq, u_eq = simulate(kap, ap_k, 40)
        t, A, _ = simulate(kap, ap_k + 0.05, T)
        a_inf = A[-1]
        shift = a_inf - ap_k
        d_inf = np.abs(A - a_inf)
        k1, c1, s1 = fit_rate(t, d_inf, 1e-7, 1e-2)
        rel.append((kap, t, d_inf, k1, c1, s1, lp_k))
        print(f"{kap:6.2f} {ap_k:7.4f} {a_inf:8.4f} {shift:11.2e} {k1:12.4f} {lp_k:14.4f} "
              f"{(k1 - lp_k) / lp_k:8.1e}   (start at a_+: a drifts to {A_eq[-1]:.4f})")

    # --- figure ----------------------------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    ax = axes[0]
    for t, A, lab in runs:
        Ap = np.where(A > 0, A, np.nan)
        line, = ax.plot(t, Ap, label=f"$a_0=${lab}")
        if A[-1] == 0:
            td = t[np.argmax(A == 0)]
            ax.plot(td, 0, "x", color=line.get_color(), ms=9)
            ax.annotate(f"dies at t = {td:.1f}", (td, 0.02), xytext=(3, 0.12), fontsize=8,
                        arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.axhline(ap, color="k", ls=":", lw=0.8, label=fr"$a_+={ap:.3f}$ (stable)")
    ax.axhline(am, color="gray", ls="--", lw=0.8, label=fr"$a_-={am:.3f}$ (unstable)")
    ax.set(xlabel="time $t$ (units of the membrane time constant)", ylabel="half-width $a(t)$",
           title=fr"Bump half-width, $\kappa={kappa}$", xlim=(0, 40))
    ax.legend(fontsize=8)

    ax = axes[1]
    for kap, t, d, k, c, s, lp_k in rel:
        line, = ax.semilogy(t, d, lw=1, label=fr"$\kappa={kap}$: fit {k:.4f}, $\lambda_e(a_+)$ = {lp_k:.4f}")
        tf = t[s]
        ax.semilogy(tf, np.exp(c + k * tf), "--", color="k", lw=0.8)
    ax.set(xlabel="time $t$", ylabel=r"$|a(t)-a(\infty)|$", ylim=(1e-9, 0.2),
           title=r"Relaxation of the wide bump, $a_0=a_++0.05$ (dashed: fits)", xlim=(0, 200))
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(X, u_eq, label=r"simulation, $\kappa=0.45$, $t=40$")
    am45, ap45, _, _ = widths(0.45)
    ax.plot(X, W(X + ap45) - W(X - ap45), "--", label=r"$U(x)=W(x+a_+)-W(x-a_+)$")
    ax.axhline(0.45, color="gray", ls=":", lw=0.8, label=r"$\kappa$")
    ax.set(xlabel="$x$", ylabel="$u(x)$", title="Stationary wide bump vs theory", xlim=(-8, 8))
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT, "problem3d_bumps.png")
    fig.savefig(path, dpi=150)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
