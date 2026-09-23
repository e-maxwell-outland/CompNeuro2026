"""Problem 5: Rheobase and the shape of the f-I curve in Hodgkin-Huxley.

Repetitive-firing criterion: a constant current Id counts as producing repetitive firing if hh_sim(Id) detects at
least 4 spikes after discarding the transient (hh_sim already drops the first `skipfrac` of the trace before counting
spikes).
"""

import numpy as np
import matplotlib.pyplot as plt

def is_repetitive(Id, gK=36.0, tmax=300.0, min_spikes=4):
    """True if constant current Id gives >= min_spikes after the transient."""
    _, _, spk, _ = hh_sim(Id=Id, gK=gK, tmax=tmax)
    return len(spk) >= min_spikes


def find_rheobase(gK=36.0, I_lo=0.0, I_hi=15.0, tol=1e-3, tmax=300.0):
    """Bisection search for the smallest Id giving repetitive firing.

    Requires the bracket to start correctly split: I_lo must NOT fire
    repetitively and I_hi MUST. tol is in the same units as Id
    (uA/cm^2); 1e-3 gives well more than the 3 significant digits asked
    for, at the cost of ~15 extra hh_sim calls (each a stiff ODE solve,
    so a bisection run takes a few seconds).
    """
    if is_repetitive(I_lo, gK, tmax):
        raise ValueError("I_lo already fires repetitively -- widen the bracket downward")
    if not is_repetitive(I_hi, gK, tmax):
        raise ValueError("I_hi does not fire repetitively -- widen the bracket upward")

    while (I_hi - I_lo) > tol:
        I_mid = 0.5 * (I_lo + I_hi)
        # Repetitive at the midpoint -> rheobase is at or below I_mid,
        # so I_mid becomes the new upper bound. Otherwise it's above.
        if is_repetitive(I_mid, gK, tmax):
            I_hi = I_mid
        else:
            I_lo = I_mid
    return 0.5 * (I_lo + I_hi)


def fI_curve(rheobase, gK=36.0, n_points=25):
    """Firing rate at a sweep of currents from just above rheobase to 2x it."""
    Id_values = np.linspace(rheobase * 1.001, rheobase * 2.0, n_points)
    rates = np.array([hh_sim(Id=Id, gK=gK)[3] for Id in Id_values])
    return Id_values, rates


if __name__ == "__main__":
    # ---------------- (a) rheobase at the default gK = 36 ----------------
    rheobase = find_rheobase(gK=36.0)
    print(f"(a) rheobase (gK=36): {rheobase:.3g} uA/cm^2")

    # Small offset so "just below" / "just above" don't straddle enough
    # of a gap to look like a different regime.
    delta = 0.01
    t_below, V_below, spk_below, _ = hh_sim(Id=rheobase - delta)
    t_above, V_above, spk_above, _ = hh_sim(Id=rheobase + delta)
    print(f"    spikes below: {len(spk_below)}, spikes above: {len(spk_above)}")

    fig, (ax_below, ax_above) = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    ax_below.plot(t_below, V_below, color="C0")
    ax_below.set_title(f"Id = {rheobase - delta:.3g} (below rheobase)")
    ax_below.set_xlabel("t (ms)")
    ax_below.set_ylabel("V (mV)")
    ax_below.set_ylim(-20, 120)

    ax_above.plot(t_above, V_above, color="C1")
    ax_above.set_title(f"Id = {rheobase + delta:.3g} (above rheobase)")
    ax_above.set_xlabel("t (ms)")

    fig.suptitle("Voltage traces just below/above rheobase")
    fig.tight_layout()
    fig.savefig("problem5a_traces.png", dpi=150)

    # ---------------- (b) f-I curve from rheobase to ~2x rheobase ----------------
    Id_values, rates = fI_curve(rheobase, gK=36.0)
    onset_rate = rates[0]
    print(f"(b) onset firing rate (just above rheobase): {onset_rate:.3g} Hz")

    fig, ax = plt.subplots()
    ax.plot(Id_values, rates, "o-")
    ax.set_xlabel("Id (uA/cm^2)")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_title("f-I curve (gK = 36)")
    fig.savefig("problem5b_fI_curve.png", dpi=150)

    # ---------------- (c) repeat with gK = 30, compare ----------------
    rheobase_gK30 = find_rheobase(gK=30.0)
    print(f"(c) rheobase: gK=36 -> {rheobase:.3g}, gK=30 -> {rheobase_gK30:.3g} uA/cm^2")

    Id_values_30, rates_30 = fI_curve(rheobase_gK30, gK=30.0)

    fig, ax = plt.subplots()
    ax.plot(Id_values, rates, "o-", color="C0", label="gK = 36")
    ax.plot(Id_values_30, rates_30, "o-", color="C1", label="gK = 30")
    ax.axvline(rheobase, color="C0", linestyle="--")
    ax.axvline(rheobase_gK30, color="C1", linestyle="--")
    ax.set_xlabel("Id (uA/cm^2)")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_title("f-I curves: gK = 36 vs gK = 30 (dashed lines = rheobase)")
    ax.legend()
    fig.savefig("problem5c_fI_comparison.png", dpi=150)

    plt.show()
