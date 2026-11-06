"""Problem 1c: Ocular dominance stripes on a ring.

Time in units of tau_w/(q_S - q_D), so part (b) reads
    dw_-(a)/dt = sum_b K(a - b) w_-(b),   K(x) = exp(-x^2/(2 s_e^2)) - (s_e/s_i) exp(-x^2/(2 s_i^2)),
on a ring of N = 200 sites with periodic distance x = min(|a-b|, N-|a-b|). Forward Euler, dt = 0.02 to t = 40,
clipping w_- to [-1, 1] (w_R, w_L in [0, w_max]) after each step. w_-(a; 0) ~ N(0, 0.01^2) i.i.d.

K is circulant, so the sum is a circular convolution: ifft(Khat * fft(w)), and the Fourier mode m (wavenumber
k = 2 pi m / N) grows at exactly the rate Khat_m = sum_x K(x) e^{-2 pi i m x/N} (real, since K is even).
From part (b):  Ktilde(k) = sqrt(2 pi) s_e [exp(-s_e^2 k^2/2) - exp(-s_i^2 k^2/2)],
                k*^2 = 4 ln(s_i/s_e)/(s_i^2 - s_e^2).
Khat_m approximates Ktilde(2 pi m/N) (lattice sum vs integral), and Ktilde(0) = 0: the uniform mode (all one
eye) does not grow.

Stripe period = N / (number of right/left stripe pairs), pairs = (sign changes around the ring)/2.

Validation (asserted before the runs): FFT convolution equals the direct circulant product, and in the linear
regime a single cosine of mode m grows by exactly (1 + dt Khat_m)^n under Euler, i.e. ~ e^{Khat_m t}.
"""

import numpy as np
import matplotlib.pyplot as plt

N, DT, T_END, SE = 200, 0.02, 40.0, 1.0


def kernel(si, se=SE, n=N):
    """K on the ring, indexed by offset 0..n-1 with periodic distance."""
    x = np.minimum(np.arange(n), n - np.arange(n))
    return np.exp(-x**2 / (2 * se**2)) - (se / si) * np.exp(-x**2 / (2 * si**2))


def k_tilde(k, si, se=SE):
    return np.sqrt(2 * np.pi) * se * (np.exp(-se**2 * k**2 / 2) - np.exp(-si**2 * k**2 / 2))


def k_star(si, se=SE):
    return np.sqrt(4 * np.log(si / se) / (si**2 - se**2))


def simulate(Khat, w0, dt=DT, t_end=T_END, clip=True):
    w = w0.copy()
    for _ in range(int(round(t_end / dt))):
        w = w + dt * np.fft.ifft(Khat * np.fft.fft(w)).real
        if clip:
            np.clip(w, -1.0, 1.0, out=w)
    return w


def stripe_pairs(w):
    s = np.sign(w)
    return np.count_nonzero(s != np.roll(s, 1)) // 2


def validate(si, rng):
    K = kernel(si)
    Khat = np.fft.fft(K).real
    assert np.abs(np.fft.fft(K).imag).max() < 1e-10
    # FFT convolution vs direct circulant product C[a, b] = K(a - b)
    C = K[(np.arange(N)[:, None] - np.arange(N)[None, :]) % N]
    w = rng.standard_normal(N)
    assert np.allclose(C @ w, np.fft.ifft(Khat * np.fft.fft(w)).real, atol=1e-12)
    # linear growth of a single mode, no clipping
    m, t = int(np.argmax(Khat[:N // 2])), 2.0
    w0 = 1e-3 * np.cos(2 * np.pi * m * np.arange(N) / N)
    g_num = simulate(Khat, w0, t_end=t, clip=False)[0] / w0[0]
    g_euler = (1 + DT * Khat[m]) ** int(round(t / DT))
    assert abs(g_num / g_euler - 1) < 1e-10
    return Khat, g_num, np.exp(Khat[m] * t)


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    n_ic = 20
    results = {}
    for si in (2.0, 3.0):
        Khat, g_num, g_exp = validate(si, rng)
        m_best = int(np.argmax(Khat[:N // 2]))
        ks = k_star(si)
        finals, periods = [], []
        for _ in range(n_ic):
            w = simulate(Khat, 0.01 * rng.standard_normal(N))
            finals.append(w)
            periods.append(N / stripe_pairs(w))
        periods = np.array(periods)
        results[si] = dict(Khat=Khat, m=m_best, finals=finals, periods=periods)

        print(f"sigma_i = {si:g}:")
        print(f"  validation: FFT conv == circulant product; Euler growth of mode {m_best} over t=2 = "
              f"{g_num:.6g} (exact (1+dt K)^n), e^(K t) = {g_exp:.6g}, rel diff {g_num/g_exp-1:.2e} "
              f"(O(dt) prediction -K^2 dt t/2 = {-Khat[m_best]**2*DT*2/2:.2e})")
        print(f"  Khat_0 = {Khat[0]:.3e} (uniform mode; continuous Ktilde(0) = 0)")
        print(f"  k* = {ks:.4f}, 2pi/k* = {2*np.pi/ks:.4f}, Ktilde(k*) = {k_tilde(ks, si):.4f}")
        print(f"  fastest ring mode m = {m_best}: period N/m = {N/m_best:.4f}, Khat_m = {Khat[m_best]:.4f}, "
              f"Ktilde(2 pi m/N) = {k_tilde(2*np.pi*m_best/N, si):.4f}")
        pairs = np.rint(N / periods).astype(int)
        print(f"  measured period over {n_ic} ICs: mean {periods.mean():.4f}, sd {periods.std(ddof=1):.4f}, "
              f"range [{periods.min():.4f}, {periods.max():.4f}]")
        u, c = np.unique(pairs, return_counts=True)
        print("  stripe-pair counts: " + ", ".join(f"{x} pairs x{y}" for x, y in zip(u, c)))
        print(f"  final |w_-| = 1 at {np.mean([np.mean(np.abs(f) == 1) for f in finals])*100:.1f}% of sites")

    # --- figure ---
    fig, ax = plt.subplots(3, 2, figsize=(12, 10))
    a = np.arange(N)
    for j, si in enumerate((2.0, 3.0)):
        r = results[si]
        ks, m = k_star(si), r["m"]
        w = r["finals"][0]
        ax[0, j].step(a, w, where="mid", color="k", lw=1)
        ax[0, j].fill_between(a, w, 0, where=w > 0, step="mid", color="tab:red", alpha=0.4, label="right eye")
        ax[0, j].fill_between(a, w, 0, where=w < 0, step="mid", color="tab:blue", alpha=0.4, label="left eye")
        ax[0, j].set_xlabel("cortical position $a$ (sites)")
        ax[0, j].set_ylabel(r"$w_-(a) = w_R - w_L$ (units of $w_{max}$)")
        ax[0, j].set_title(rf"final pattern at $t=40\,\tau_w/(q_S-q_D)$, $\sigma_e=1$, $\sigma_i={si:g}$, "
                           rf"period {N/stripe_pairs(w):.3f}")
        ax[0, j].legend(fontsize=8, loc="upper right")

        # pair counts are integers, so plot those; the top axis gives the period N/pairs of each bar
        p = r["periods"]
        pairs = np.rint(N / p).astype(int)
        n_cont = N * ks / (2 * np.pi)
        ticks = np.arange(min(pairs.min(), int(n_cont)) - 1, max(pairs.max(), m) + 2)
        counts = [np.count_nonzero(pairs == n) for n in ticks]
        bars = ax[1, j].bar(ticks, counts, width=0.8, color="gray", zorder=2,
                            label="simulation")
        ax[1, j].bar_label(bars, labels=[str(c) if c else "" for c in counts], fontsize=8, padding=2,
                           bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"), zorder=4)
        ax[1, j].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax[1, j].axvline(n_cont, color="r", ls="--", zorder=1,
                         label=rf"continuum $2\pi/k_*$: {n_cont:.1f} pairs")
        ax[1, j].axvline(m, color="b", ls=":", lw=2, zorder=1,
                         label=f"fastest ring mode: {m} pairs")
        ax[1, j].errorbar(pairs.mean(), max(counts) + 1.5, xerr=pairs.std(ddof=1), fmt="kv", ms=7, capsize=4,
                          label=rf"mean $\pm$ sd: {pairs.mean():.1f} $\pm$ {pairs.std(ddof=1):.1f} pairs")
        ax[1, j].set_xticks(ticks)
        ax[1, j].set_xlabel("number of right/left stripe pairs on the ring")
        ax[1, j].set_ylabel("number of runs")
        ax[1, j].set_ylim(0, max(counts) * 2.1)
        top = ax[1, j].secondary_xaxis("top")
        top.set_xticks(ticks, labels=[f"{N/n:.2f}" for n in ticks], fontsize=8)
        top.set_xlabel(r"stripe period $N/\mathrm{pairs}$ (sites)", fontsize=9)
        ax[1, j].set_title(rf"stripe count over {len(p)} initial conditions, $\sigma_i={si:g}$", pad=8)
        ax[1, j].legend(fontsize=7.5, loc="upper left" if j == 0 else "upper right", framealpha=1).set_zorder(5)

        mm = np.arange(N // 2 + 1)
        kc = np.linspace(0, np.pi, 400)
        ax[2, j].plot(kc, k_tilde(kc, si), "r", lw=1.5, label=r"continuous $\tilde K(k)$")
        ax[2, j].plot(2 * np.pi * mm / N, r["Khat"][:N // 2 + 1], "k.", ms=3, label=r"ring modes $\hat K_m$")
        ax[2, j].axvline(ks, color="r", ls="--", lw=1, label=rf"$k_*={ks:.3f}$")
        ax[2, j].axhline(0, color="gray", lw=0.8)
        ax[2, j].set_xlabel(r"wavenumber $k$ (rad/site)")
        ax[2, j].set_ylabel(r"growth rate (units of $(q_S-q_D)/\tau_w$)")
        ax[2, j].set_title(rf"growth spectrum, $\sigma_i={si:g}$")
        ax[2, j].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("problem1c_stripes.png", dpi=150)
