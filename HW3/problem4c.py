"""Problem 4c: Gap-junction coupled QIF neurons, simulated against the phase reduction.

    dv1/dt = 1 + v1^2 + eps (v2 - v1),   dv2/dt = 1 + v2^2 + eps (v1 - v2),
    threshold +L, reset -L, L = 200 (standing in for +-infinity)

From parts a-b: H(chi) = sin(2 pi chi)/(2 pi), so phi = theta1 - theta2 obeys phi' = -(eps/pi) sin(2 pi phi),
whose solution is
    tan(pi phi(t)) = tan(pi phi0) e^{-2 eps t}.
Synchrony is stable, and the synchronous frequency is unchanged (H(0) = 0).

Neuron 1 starts at reset, and neuron 2 starts phi0 = 0.4 of a cycle behind on the uncoupled cutoff orbit.
With the cutoff the period is Delta_L = 2 arctan L = pi - 2/L + ..., not exactly pi, so neuron 2 is placed
at v2 = tan(arctan L - phi0 Delta_L). Then it spikes at phi0 Delta_L, phi0 of a cycle after neuron 1's
(virtual) previous spike. This is -cot(0.6 pi) = 0.325 to within O(1/L).

Lag per cycle: phi_k = (t2_k - t1_k)/(t1_{k+1} - t1_k), where t2_k is the first neuron-2 spike after t1_k.
Spikes are terminal events of solve_ivp, which stops, resets the neuron to -L and restarts.

Starting neuron 1 AT reset breaks that cancellation once. Neuron 2 sees only the -L half of neuron 1's
spike (the +L half "before" t = 0 never happened), so it gets a one-time kick
    dv2 = eps int v1 dt ~ -eps log L,
which delays it by R(1 - phi0) eps log L = eps log(L) sin^2(pi(1 - phi0))/pi cycles (0.031 at eps = 0.02).
After that first cycle every partner spike is complete. So the prediction is also drawn started from the
first measured lag (t1_0, phi_0) instead of from t = 0; that sidesteps the start-up kick without modeling it.

Numerical notes. Each run lasts 12/(2 eps), so tan(pi phi) falls by about e^-12 and the lag reaches ~1e-5.
Decay rates are fit over the whole run and over the near-synchronous part (phi < 0.01) alone. Near synchrony
the partner spikes when R(theta) ~ 0, so the finite cutoff hardly matters, and the rate is within about 0.5%
of 2 eps. At large lags the partner spikes mid-cycle where R is large, and the cutoff and O(eps) corrections
pull the rate a few % low. Both shrink as L grows and eps shrinks.

Why the cutoff must be symmetric: over one cutoff cycle the time average of the partner's voltage is
    <v> = (1/Delta) int_{-L_r}^{L_t} v/(1 + v^2) dv = log[(1 + L_t^2)/(1 + L_r^2)] / (2 Delta).
With L_t = L_r this is zero. That is the finite-L version of the principal value in part b: the partner's
excursion to +L just before its spike cancels its excursion to -L just after. An asymmetric cutoff leaves
a net voltage pulse at every partner spike, which acts like a spurious extra synapse that H(chi) does not contain.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def simulate_pair(eps, L, phi0, t_end):
    """Spike times of both neurons on [0, t_end]."""
    def rhs(t, y):
        return [1 + y[0]**2 + eps * (y[1] - y[0]), 1 + y[1]**2 + eps * (y[0] - y[1])]

    events = []
    for i in range(2):
        ev = lambda t, y, i=i: y[i] - L
        ev.terminal, ev.direction = True, 1
        events.append(ev)

    D_L = 2 * np.arctan(L)
    y = np.array([-L, np.tan(np.arctan(L) - phi0 * D_L)])
    t, spikes = 0.0, ([], [])
    while t < t_end:
        sol = solve_ivp(rhs, [t, t_end], y, method="DOP853", events=events, rtol=1e-10, atol=1e-10)
        t, y = sol.t[-1], sol.y[:, -1].copy()
        for i in range(2):
            if sol.t_events[i].size:
                spikes[i].append(sol.t_events[i][0])
                y[i] = -L
    return np.array(spikes[0]), np.array(spikes[1])


def lags(s1, s2):
    """phi_k and t1_k for every neuron-1 cycle with a neuron-2 spike inside it."""
    t1, phi = [], []
    for k in range(len(s1) - 1):
        j = np.searchsorted(s2, s1[k], side="right")
        if j < len(s2) and s2[j] < s1[k + 1]:
            t1.append(s1[k])
            phi.append((s2[j] - s1[k]) / (s1[k + 1] - s1[k]))
    return np.array(t1), np.array(phi)


def predicted_lag(t, eps, phi0):
    return np.arctan(np.tan(np.pi * phi0) * np.exp(-2 * eps * t)) / np.pi


if __name__ == "__main__":
    L, phi0 = 200.0, 0.4
    D_L = 2 * np.arctan(L)

    # --- control: eps = 0 keeps the initial lag and the cutoff period ---
    s1, s2 = simulate_pair(0.0, L, phi0, 40.0)
    _, ph = lags(s1, s2)
    print(f"check eps=0: lags in [{ph.min():.5f}, {ph.max():.5f}] (should be {phi0}), "
          f"ISI {np.diff(s1).mean():.5f} (Delta_L = 2 arctan L = {D_L:.5f})")

    print(f"\nL = {L:g}, phi0 = {phi0}")
    print(f"{'eps':>6} {'2 eps':>7} {'rate (all)':>10} {'rel err':>8} {'rate (phi<0.01)':>15} {'rel err':>8} "
          f"{'cycles':>6} {'last lag':>9} | "
          f"{'ISI 1':>8} {'ISI 2':>8} {'pair mean - Delta_L':>20}")
    runs = {}
    for eps in [0.005, 0.02, 0.05]:
        s1, s2 = simulate_pair(eps, L, phi0, 12 / (2 * eps))
        t1, ph = lags(s1, s2)
        y = np.log(np.tan(np.pi * ph))
        slope = np.polyfit(t1, y, 1)[0]
        m = ph < 0.01
        slope_sync = np.polyfit(t1[m], y[m], 1)[0]
        runs[eps] = (t1, ph, slope)
        print(f"{eps:>6g} {2*eps:>7.4f} {-slope:>10.5f} {(-slope - 2*eps)/(2*eps):>8.2%} "
              f"{-slope_sync:>15.5f} {(-slope_sync - 2*eps)/(2*eps):>8.2%} {len(ph):>6d} {ph[-1]:>9.1e} | "
              f"{np.diff(s1).mean():>8.5f} {np.diff(s2).mean():>8.5f} "
              f"{(np.diff(s1).mean() + np.diff(s2).mean()) / 2 - D_L:>20.1e}")
    print(f"Delta_L = {D_L:.5f}. While lagged the leader (1) slows and the follower (2) speeds up equally,")
    print("so the pair-mean period moves only at O(eps^2). Near synchrony the rate matches 2 eps to ~0.5%;")
    print("at large lags the finite cutoff and O(eps) corrections pull it a few % low (see docstring).")

    t1, ph, _ = runs[0.02]
    print(f"\neps = 0.02: max |phi_k - prediction|: from phi0 = {phi0} at t = 0: "
          f"{np.abs(ph - predicted_lag(t1, 0.02, phi0)).max():.4f};  from first measured lag "
          f"{ph[0]:.4f} at t = {t1[0]:.3f}: {np.abs(ph - predicted_lag(t1 - t1[0], 0.02, ph[0])).max():.4f}")
    print(f"  first lag {ph[0]:.4f} > {phi0}: the start-up half-spike kick predicts "
          f"+eps log(L) sin^2(pi(1-phi0))/pi = +{0.02*np.log(L)*np.sin(np.pi*(1-phi0))**2/np.pi:.4f}")

    for Lt, Lr in [(L, L), (L, L / 2)]:
        D = np.arctan(Lt) + np.arctan(Lr)
        print(f"cycle-mean of partner voltage, cutoff +{Lt:g}/-{Lr:g}: "
              f"{np.log((1 + Lt**2) / (1 + Lr**2)) / (2 * D):.4f}")

    # --- figure ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    tc = np.linspace(0, t1[-1], 500)
    ax[0].plot(tc, predicted_lag(tc, 0.02, phi0), "r", lw=2,
               label=r"phase reduction $\tan\pi\phi=\tan(\pi\phi_0)e^{-2\epsilon t}$")
    tc1 = tc[tc >= t1[0]]
    ax[0].plot(tc1, predicted_lag(tc1 - t1[0], 0.02, ph[0]), "b--", lw=1.5,
               label="phase reduction restarted at first measured lag\n"
                     rf"$\tan\pi\phi=\tan(\pi\cdot{ph[0]:.3f})\,e^{{-2\epsilon(t-{t1[0]:.2f})}}$")
    ax[0].plot(t1, ph, "k.", ms=6, label=r"simulation $\phi_k$")
    ax[0].set_xlabel(r"time $t$ of neuron-1 spike $t^{(1)}_k$ (model time units)")
    ax[0].set_ylabel(r"lag $\phi_k$ (fraction of a cycle)")
    ax[0].set_title(rf"QIF pair, gap junction $\epsilon=0.02$, $L={L:g}$, $\phi_0={phi0}$")
    ax[0].legend(fontsize=9)

    for (eps, (t1, ph, slope)), c in zip(runs.items(), ["C0", "C1", "C2"]):
        ax[1].plot(t1, np.log(np.tan(np.pi * ph)), ".", color=c, ms=4,
                   label=rf"$\epsilon={eps:g}$: fit rate {-slope:.4f} vs $2\epsilon={2*eps:g}$")
        ax[1].plot(t1, np.log(np.tan(np.pi * ph[0])) - 2 * eps * (t1 - t1[0]), "--", color=c, lw=1)
    ax[1].set_xlabel(r"time $t^{(1)}_k$ (model time units)")
    ax[1].set_ylabel(r"$\log\tan(\pi\phi_k)$")
    ax[1].set_title(r"decay of the lag; dashed: slope $-2\epsilon$ from first measured lag")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("problem4c_phase_lag.png", dpi=150)
