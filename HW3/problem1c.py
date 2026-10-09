"""Problem 1c: Mode-locking of the periodically forced LIF neuron

    dv/dt = -v + I + eps sin(2 pi t),   v -> 0 when v = 1,   forcing period 1

From part a, the steady response to the forcing is
    G(t) = int_{-inf}^0 e^s I(t+s) ds = I + eps (sin 2 pi t - 2 pi cos 2 pi t)/(1 + 4 pi^2)
and between spikes v(t) = G(t) + [v(T_n) - G(T_n)] e^{-(t - T_n)} exactly, with v(T_n) = 0 after reset.

So the simulation is event driven: it only has to find the first time v reaches 1 after each reset.

Each crossing is bracketed on a grid of spacing h = 0.005 and refined by brentq, so nothing is discretized.

Predictions from parts a-b:
    1:1 tongue  |I - I*| <= eps/sqrt(1 + 4 pi^2),   I* = 1/(1 - e^-1)
    locked phases solve G(phi) = I*, with multiplier lambda = e^-1 I(phi)/(I(phi) - 1);
    for I = 1.6, eps = 0.5 the stable one is phi = 0.1881 (lambda = 0.714).

Unforced firing number (eps = 0): 1/log[I/(I - 1)] for I > 1, zero otherwise.
"""

import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
from scipy.integrate import quad
from scipy.optimize import brentq

TWOPI = 2 * np.pi
I_STAR = 1 / (1 - np.exp(-1))


def G(t, I, eps):
    return I + eps * (np.sin(TWOPI * t) - TWOPI * np.cos(TWOPI * t)) / (1 + 4 * np.pi**2)


def next_spike(Tn, I, eps, t_end, h=0.005, window=4.0):
    """First t > Tn with v(t) = 1 after a reset at Tn, or None if none before t_end."""
    GTn = G(Tn, I, eps)
    v = lambda t: G(t, I, eps) - GTn * np.exp(-(t - Tn)) - 1
    t0 = Tn
    while t0 < t_end:
        t = t0 + h * np.arange(1, int(window / h) + 1)
        above = np.nonzero(v(t) >= 0)[0]
        if above.size:
            i = above[0]
            lo = t[i - 1] if i > 0 else t0
            return brentq(v, lo, t[i], xtol=1e-13)
        t0 = t[-1]
    return None


def spike_train(I, eps, t_end, t0=0.0):
    """All spike times on (t0, t_end], starting from a reset v(t0) = 0."""
    spikes, Tn = [], t0
    while True:
        Tn = next_spike(Tn, I, eps, t_end)
        if Tn is None or Tn > t_end:
            return np.array(spikes)
        spikes.append(Tn)


def spike_count(I, eps, n_trans=100, n_meas=300):
    """Spikes in the n_meas forcing periods after an n_trans period transient."""
    s = spike_train(I, eps, n_trans + n_meas)
    return np.count_nonzero(s > n_trans)


def unforced_firing_number(I):
    I = np.asarray(I, dtype=float)
    out = np.zeros_like(I)
    m = I > 1
    out[m] = 1 / np.log(I[m] / (I[m] - 1))
    return out


def locked_phases(I, eps, n=4001):
    """Roots of G(phi) = I* on [0, 1) and their multipliers lambda."""
    f = lambda p: G(p, I, eps) - I_STAR
    xs = np.linspace(0, 1, n)
    fx = f(xs)
    roots = [brentq(f, xs[i], xs[i + 1]) for i in range(n - 1) if fx[i] * fx[i + 1] < 0]
    Iphi = [I + eps * np.sin(TWOPI * p) for p in roots]
    return [(p, np.exp(-1) * Ip / (Ip - 1)) for p, Ip in zip(roots, Iphi)]


def euler_spikes(I, eps, T, dt):
    """Brute-force forward Euler with reset, for checking the event-driven solver."""
    v, spikes = 0.0, []
    for j in range(int(round(T / dt))):
        t = j * dt
        v += dt * (-v + I + eps * np.sin(TWOPI * t))
        if v >= 1:
            v = 0.0
            spikes.append(t + dt)
    return np.array(spikes)


if __name__ == "__main__":
    eps = 0.5
    n_trans, n_meas = 100, 300
    half = eps / np.sqrt(1 + 4 * np.pi**2)
    lo_edge, hi_edge = I_STAR - half, I_STAR + half

    # --- check 1: closed-form G against quadrature of its definition ---
    err = max(abs(G(t, 1.6, eps) - quad(lambda s: np.exp(s) * (1.6 + eps * np.sin(TWOPI * (t + s))),
                                        -60, 0, limit=500)[0]) for t in [0.0, 0.13, 0.37, 0.71])
    print(f"check G closed form vs quadrature: max error {err:.1e}")

    # --- check 2: eps = 0 reproduces the unforced firing number ---
    # grid avoids I = 1 exactly, where 1 - e^-t rounds to 1 in floating point and fakes spikes
    Ichk = np.arange(0.95, 3, 0.1)
    fn0 = np.array([spike_count(I, 0.0, n_trans, n_meas) for I in Ichk]) / n_meas
    print(f"check eps=0: max |firing number - 1/log(I/(I-1))| = "
          f"{np.abs(fn0 - unforced_firing_number(Ichk)).max():.4f}  (resolution 1/{n_meas} = {1/n_meas:.4f})")

    # --- check 3: event-driven spike times against brute-force Euler ---
    for I in [1.6, 2.5]:
        se, sb = spike_train(I, eps, 10.0), euler_spikes(I, eps, 10.0, 1e-5)
        k = min(len(se), len(sb))
        print(f"check Euler dt=1e-5, I={I}: {len(se)} vs {len(sb)} spikes on [0,10], "
              f"max |dT| = {np.abs(se[:k] - sb[:k]).max():.1e}")

    # --- parts a-b at I = 1.6: predicted locked states vs simulation ---
    I0 = 1.6
    print(f"\nI* = {I_STAR:.5f}, tongue half-width eps/sqrt(1+4pi^2) = {half:.5f}, "
          f"1:1 tongue I in [{lo_edge:.4f}, {hi_edge:.4f}]")
    for p, lam in locked_phases(I0, eps):
        print(f"  predicted locked phase phi = {p:.4f}, lambda = {lam:.4f} -> "
              f"{'stable' if abs(lam) < 1 else 'unstable'}")
    s = spike_train(I0, eps, 200.0)
    ph = s % 1
    phi_s = locked_phases(I0, eps)[0][0]
    print(f"  simulated spike phase at I = {I0} (last 5): {np.round(ph[-5:], 4)}, "
          f"spikes/period = {np.count_nonzero(s > 100) / 100:.3f}")
    d = (ph - phi_s + 0.5) % 1 - 0.5
    good = np.nonzero((np.abs(d[:-1]) > 1e-7) & (np.abs(d[:-1]) < 1e-2))[0]
    print(f"  measured contraction delta_(n+1)/delta_n = {np.median(d[good + 1] / d[good]):.4f} "
          f"(predicted {locked_phases(I0, eps)[0][1]:.4f})")

    # --- start at the unstable phase phi2, nudged by +-1e-6: it should leave at lambda2 and land on phi1 ---
    (phi1, _), (phi2, lam2) = locked_phases(I0, eps)
    wrap = lambda x: (x + 0.5) % 1 - 0.5
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for off, c in [(1e-6, "C0"), (-1e-6, "C3")]:
        T0 = phi2 + off
        ph = np.concatenate([[T0], spike_train(I0, eps, 60.0, t0=T0)]) % 1
        d2 = wrap(ph - phi2)
        g = np.nonzero((np.abs(d2[:-1]) > 1e-7) & (np.abs(d2[:-1]) < 1e-2))[0]
        print(f"  start at phi2 {off:+.0e}: measured growth delta_(n+1)/delta_n = "
              f"{np.median(d2[g + 1] / d2[g]):.4f} (predicted {lam2:.4f}), final phase {ph[-1]:.4f}")
        n = np.arange(len(ph))
        ax.plot(n, ph, ".-", color=c, ms=5, lw=0.8, label=rf"reset at $\phi_2{off:+.0e}$")
    for p, lab in [(phi1, rf"$\phi_1={phi1:.4f}$ (stable)"), (phi2, rf"$\phi_2={phi2:.4f}$ (unstable)")]:
        ax.axhline(p, color="gray", ls="--", lw=1)
        ax.text(60, p + 0.02, lab, ha="right", fontsize=9)
    ax.set_xlabel(r"spike number $n$")
    ax.set_ylabel(r"spike phase $\phi_n = T_n\ \mathrm{mod}\ 1$")
    ax.set_title(rf"Leaving the unstable 1:1 state, $I={I0}$, $\epsilon={eps}$")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="center right")
    fig.tight_layout()
    fig.savefig("problem1c_unstable.png", dpi=150)

    # --- the sweep ---
    Is = np.round(np.arange(0.9, 3.0 + 1e-9, 0.002), 4)
    counts = np.array([spike_count(I, eps, n_trans, n_meas) for I in Is])
    fn = counts / n_meas

    # plateaus: runs of >= 3 grid points with the same count, equal to p/q with q <= 6
    plateaus, i = [], 0
    while i < len(Is):
        j = i
        while j + 1 < len(Is) and counts[j + 1] == counts[i]:
            j += 1
        r = Fraction(int(counts[i]), n_meas).limit_denominator(6)
        if j - i >= 2 and r * n_meas == counts[i]:
            plateaus.append((r, Is[i], Is[j]))
        i = j + 1
    print(f"\nplateaus (eps = {eps}, I step 0.002, {n_meas} periods after {n_trans}):")
    print(f"{'p:q':>5} {'firing #':>8} {'I from':>7} {'I to':>7} {'width':>6}")
    for r, a, b in plateaus:
        print(f"{r.numerator:>2}:{r.denominator:<2} {float(r):>8.4f} {a:>7.3f} {b:>7.3f} {b - a:>6.3f}")
    one = [(a, b) for r, a, b in plateaus if r == 1][0]
    print(f"1:1 plateau measured [{one[0]:.3f}, {one[1]:.3f}] vs tongue [{lo_edge:.4f}, {hi_edge:.4f}]")

    # --- figure ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    Ic = np.linspace(0.9, 3, 2000)
    ax.plot(Is, fn, "k-", lw=1.2, label=rf"forced, $\epsilon={eps}$ (simulation)")
    ax.plot(Ic, unforced_firing_number(Ic), "b--", lw=1.2, label=r"unforced $1/\log[I/(I-1)]$")
    ax.axvspan(lo_edge, hi_edge, color="r", alpha=0.15,
               label=rf"1:1 tongue $|I-I^*|\leq\epsilon/\sqrt{{1+4\pi^2}}$")
    for x in (lo_edge, hi_edge):
        ax.axvline(x, color="r", ls=":", lw=1)
    # label the 1:1 plateau plus the four others reported in the write-up
    labeled = {Fraction(1, 3), Fraction(1, 2), Fraction(2, 3), Fraction(1), Fraction(2)}
    for r, a, b in plateaus:
        if r in labeled:
            ax.annotate(f"{r.numerator}:{r.denominator}", ((a + b) / 2, float(r)),
                        xytext=(-14, 6), textcoords="offset points", fontsize=8)
    ax.set_xlabel(r"$I$")
    ax.set_ylabel("firing number (spikes per forcing period)")
    ax.set_title(rf"Forced LIF $\dot v=-v+I+\epsilon\sin 2\pi t$, $\epsilon={eps}$, "
                 rf"{n_meas} periods after {n_trans}-period transient")
    ax.legend(fontsize=9, loc="upper left")

    ins = ax.inset_axes([0.6, 0.1, 0.36, 0.33])
    m = (Is > 1.45) & (Is < 1.72)
    ins.plot(Is[m], fn[m], "k.-", ms=3, lw=0.8)
    ins.plot(Ic[(Ic > 1.45) & (Ic < 1.72)], unforced_firing_number(Ic[(Ic > 1.45) & (Ic < 1.72)]), "b--", lw=1)
    ins.axvspan(lo_edge, hi_edge, color="r", alpha=0.15)
    ins.set_title("zoom on 1:1 plateau", fontsize=8)
    ins.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig("problem1c_firing_number.png", dpi=150)
