"""Problem 2d: Binocular rivalry with slow adaptation, smooth sigmoid version.

    du_L/dt = -u_L + f(w u_L - beta u_R - g a_L + I_L),   tau_a da_L/dt = -a_L + u_L,
    du_R/dt = -u_R + f(w u_R - beta u_L - g a_R + I_R),   tau_a da_R/dt = -a_R + u_R,
f(x) = 1/(1 + e^{-eta x}), eta = 1000, w = 0.8, beta = 1, g = 2, tau_a = 120 (time in units of the rate constant).

Integration: solve_ivp Radau (stiff; analytic Jacobian), rtol 1e-9, atol 1e-11, t in [0, 30 tau_a], from
(u_L, u_R, a_L, a_R) = (1, 0, 0, 0). The solution is sampled every 0.01 from the dense output.
Dominance: L is dominant when u_L - u_R > 1/2, R when u_L - u_R < -1/2. A hand-over is the time the new winner
first crosses its threshold (hysteresis: intervals where |u_L - u_R| < 1/2 never count as a switch). The
durations between hand-overs are attributed to the percept that held during them; the first three are discarded.

Predictions (Heaviside limit, parts b, c, and the release formula):
    B_j = (I_j - beta)/g,  T_L = tau_a ln((1 - B_L)/B_R),  T_R = tau_a ln((1 - B_R)/B_L)   (escape),
    w = 0.3 < g + beta - 2I: the winner releases itself at a = (w+I)/g,  T_dom = tau_a ln[(w+I)/(g-w-I)].

Validation:
  * in the symmetric run the adaptation variables must swing between about B and 1-B (escape), and for w = 0.3
    between 1-(w+I)/g and (w+I)/g (release);
  * the symmetric run must give T_L = T_R;
  * rerunning (ii) at rtol 1e-11 must leave every mean duration unchanged to well below 3 significant figures.

Sign of the discrepancies: two finite-size effects compete. (1) With a finite eta the loser's low state vanishes in a
saddle-node slightly before its a reaches B (min a ~ 0.153 instead of 0.150), so it escapes early: durations shorter,
an O(ln(eta)/eta) effect. (2) The rates swap on the unit time scale, not instantly, which lengthens durations by an
O(1/tau_a) amount. The probe at the end separates them: eta = 1e4 (removes 1) gives durations longer than predicted;
tau_a = 1200 (shrinks 2) gives durations shorter than predicted.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import expit

W, BETA, G, TAU_A, ETA = 0.8, 1.0, 2.0, 120.0, 1000.0
T_END = 30 * TAU_A
OUT = os.path.dirname(os.path.abspath(__file__))


def rhs(t, y, w, IL, IR):
    uL, uR, aL, aR = y
    return [-uL + expit(ETA * (w * uL - BETA * uR - G * aL + IL)),
            -uR + expit(ETA * (w * uR - BETA * uL - G * aR + IR)),
            (-aL + uL) / TAU_A,
            (-aR + uR) / TAU_A]


def jac(t, y, w, IL, IR):
    uL, uR, aL, aR = y
    sL = expit(ETA * (w * uL - BETA * uR - G * aL + IL))
    sR = expit(ETA * (w * uR - BETA * uL - G * aR + IR))
    dL, dR = ETA * sL * (1 - sL), ETA * sR * (1 - sR)
    return np.array([[-1 + dL * w, -dL * BETA, -dL * G, 0],
                     [-dR * BETA, -1 + dR * w, 0, -dR * G],
                     [1 / TAU_A, 0, -1 / TAU_A, 0],
                     [0, 1 / TAU_A, 0, -1 / TAU_A]])


def simulate(IL, IR, w=W, rtol=1e-9, dt_s=0.01):
    sol = solve_ivp(rhs, (0, T_END), [1, 0, 0, 0], args=(w, IL, IR), method="Radau", jac=jac,
                    rtol=rtol, atol=rtol * 1e-2, dense_output=True, max_step=1.0)
    assert sol.success, sol.message
    t = np.arange(0, T_END, dt_s)
    return t, sol.sol(t)


def dominance(t, Y, n_discard=3):
    """Hand-over times, who won each, and the durations (after discarding the first n_discard)."""
    d = Y[0] - Y[1]
    state, switches = 0, []  # state: +1 L, -1 R, 0 none yet
    for i in range(1, len(t)):
        if state != 1 and d[i] > 0.5:
            # linear interpolation of the crossing
            tc = t[i - 1] + (0.5 - d[i - 1]) / (d[i] - d[i - 1]) * (t[i] - t[i - 1])
            switches.append((tc, 1, i))
            state = 1
        elif state != -1 and d[i] < -0.5:
            tc = t[i - 1] + (-0.5 - d[i - 1]) / (d[i] - d[i - 1]) * (t[i] - t[i - 1])
            switches.append((tc, -1, i))
            state = -1
    times = np.array([s[0] for s in switches])
    who = np.array([s[1] for s in switches])
    durs = np.diff(times)
    winners = who[:-1]
    return times, who, [s[2] for s in switches], durs[n_discard:], winners[n_discard:]


def predicted(IL, IR):
    BL, BR = (IL - BETA) / G, (IR - BETA) / G
    return TAU_A * np.log((1 - BL) / BR), TAU_A * np.log((1 - BR) / BL)


def mean_TL_TR(IL, IR, **kw):
    t, Y = simulate(IL, IR, **kw)
    _, _, _, durs, win = dominance(t, Y)
    return durs[win == 1].mean(), durs[win == -1].mean(), (win == 1).sum(), (win == -1).sum()


def main():
    global ETA, TAU_A, T_END
    I = 1.3
    Bsym = (I - BETA) / G

    # --- (i) symmetric run ---------------------------------------------------------------------------------------
    t, Y = simulate(I, I)
    times, who, idx, durs, win = dominance(t, Y)
    print(f"(i) I_L = I_R = {I}: B = {Bsym:.4f}, 1-B = {1 - Bsym:.4f}, T_dom(b) = {predicted(I, I)[0]:.2f}")
    print(f"    {len(times)} hand-overs; first at t = {times[:4].round(1)}")
    half = len(t) // 2
    a_lo, a_hi = Y[2:, half:].min(axis=1), Y[2:, half:].max(axis=1)
    print(f"    adaptation extrema (2nd half): min a_L, a_R = {a_lo.round(4)} (B = {Bsym:.4f}); "
          f"max a_L, a_R = {a_hi.round(4)} (1-B = {1 - Bsym:.4f})")
    TL, TR = durs[win == 1].mean(), durs[win == -1].mean()
    print(f"    T_L = {TL:.2f} (n={np.sum(win == 1)}), T_R = {TR:.2f} (n={np.sum(win == -1)}), "
          f"spread {durs.std():.2e}\n")

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    tt = t / TAU_A
    axes[0].plot(tt, Y[0], label=r"$u_L$ (vertical)")
    axes[0].plot(tt, Y[1], label=r"$u_R$ (horizontal)", ls="--")
    axes[0].set(ylabel="rate $u$", title=fr"(i) Rivalry, $I_L=I_R={I}$, $w={W}$, $\beta={BETA}$, $g={G}$, "
                                         fr"$\tau_a={TAU_A:.0f}$, $\eta={ETA:.0f}$")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].plot(tt, Y[2], label=r"$a_L$")
    axes[1].plot(tt, Y[3], label=r"$a_R$", ls="--")
    axes[1].axhline(Bsym, color="k", lw=0.8, ls=":", label=fr"$B={Bsym:.2f}$")
    axes[1].axhline(1 - Bsym, color="gray", lw=0.8, ls="-.", label=fr"$1-B={1 - Bsym:.2f}$")
    axes[1].set(xlabel=r"time $t/\tau_a$", ylabel="adaptation $a$", ylim=(-0.02, 1.05))
    axes[1].legend(loc="upper right", fontsize=8, ncol=4)
    fig.tight_layout()
    path = os.path.join(OUT, "problem2d_rivalry.png")
    fig.savefig(path, dpi=150)
    print(f"saved {path}\n")

    # --- (ii) asymmetric drive ------------------------------------------------------------------------------------
    print("(ii) I_R = 1.3")
    print(f"{'I_L':>5} {'T_L sim':>8} {'T_L (c)':>8} {'diff':>7} {'T_R sim':>8} {'T_R (c)':>8} {'diff':>7} "
          f"{'frac L sim':>10} {'frac L (c)':>10} {'nL,nR':>6}  T at rtol 1e-11")
    for IL in [1.3, 1.45, 1.6]:
        TLs, TRs, nL, nR = mean_TL_TR(IL, I)
        TLp, TRp = predicted(IL, I)
        TLf, TRf, _, _ = mean_TL_TR(IL, I, rtol=1e-11)
        print(f"{IL:5.2f} {TLs:8.1f} {TLp:8.1f} {TLs - TLp:7.2f} {TRs:8.1f} {TRp:8.1f} {TRs - TRp:7.2f} "
              f"{TLs / (TLs + TRs):10.4f} {TLp / (TLp + TRp):10.4f} {nL:>3},{nR:<3} "
              f"({TLf:.3f}, {TRf:.3f}) vs ({TLs:.3f}, {TRs:.3f})")

    # --- (iii) w = 0.3, release ----------------------------------------------------------------------------------
    w3 = 0.3
    t3, Y3 = simulate(I, I, w=w3)
    times3, who3, idx3, durs3, win3 = dominance(t3, Y3)
    Tpred = TAU_A * np.log((w3 + I) / (G - w3 - I))
    h3 = len(t3) // 2
    print(f"\n(iii) w = {w3}: T_dom sim = {durs3.mean():.1f} (n={len(durs3)}, T_L={durs3[win3 == 1].mean():.1f}, "
          f"T_R={durs3[win3 == -1].mean():.1f}); prediction tau_a ln[(w+I)/(g-w-I)] = {Tpred:.1f}; "
          f"diff {durs3.mean() - Tpred:.2f}")
    print(f"      adaptation range (2nd half): [{Y3[2:, h3:].min():.4f}, {Y3[2:, h3:].max():.4f}]; release predicts "
          f"[1-(w+I)/g, (w+I)/g] = [{1 - (w3 + I) / G:.4f}, {(w3 + I) / G:.4f}] (escape would need B = {Bsym:.2f})")

    # --- sign of the discrepancies: separate the two finite-size effects -----------------------------------------
    # eta -> larger isolates the finite-switching-time effect; tau_a -> larger isolates the sigmoid-threshold effect.
    print("\nDiscrepancy probe: simulated / predicted durations, I_R = 1.3")
    for eta, tau in [(1000.0, 120.0), (10000.0, 120.0), (1000.0, 1200.0)]:
        ETA, TAU_A, T_END = eta, tau, 30 * tau
        for IL in [1.3, 1.6]:
            tp, Yp = simulate(IL, I, dt_s=0.01 * tau / 120)
            _, _, _, dp, wp = dominance(tp, Yp)
            TLp, TRp = predicted(IL, I)
            print(f"   eta={eta:6.0f} tau_a={tau:5.0f} I_L={IL}: T_L ratio {dp[wp == 1].mean() / TLp:.4f}, "
                  f"T_R ratio {dp[wp == -1].mean() / TRp:.4f}, min a_R = {Yp[3, len(tp) // 2:].min():.4f}")
    ETA, TAU_A, T_END = 1000.0, 120.0, 30 * 120.0

if __name__ == "__main__":
    main()
