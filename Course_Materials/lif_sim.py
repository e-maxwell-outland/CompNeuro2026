"""Leaky integrate-and-fire neuron, with arbitrary input current and
optional white noise.

    tau_m du/dt = -(u - u_rest) + R I(t) + noise,   reset u -> u_rest at u_th

Consolidates lif_mod.py, lif_per.py, lif_perforce.py, and the three
integration loops in LIF_Model.ipynb and the two in Noisy_IF.ipynb, which
were all the same Euler scheme with different I(t).

From the LIF_Model notes: the model is an RC circuit. Charge leaks across
the membrane with time constant tau_m, injected current drives u upward,
and the spike itself is not modeled -- reaching u_th is declared a spike
and u is reset. Everything interesting is in what I(t) does and in when
the threshold is crossed.

Fixed from the originals: those initialized the spike list as the scalar 0,
which left a phantom spike at t = 0 and corrupted any rate or ISI estimate
built from it. Here spikes starts empty.

Regression check: with the defaults (constant I = 1.1, tau_m = 1, R = 1,
u_th = 1) the neuron fires periodically with ISI = tau_m*log(R*I/(R*I-u_th))
= 2.398 ms. If you get 0 spikes, R*I is below threshold.
"""

import numpy as np

__all__ = ["lif_sim"]


def lif_sim(I=1.1, T=10.0, dt=1e-3, taum=1.0, R=1.0, urest=0.0, uth=1.0,
            sigma=0.0, seed=None):
    """Integrate the LIF model by Euler-Maruyama.

    I : constant, or a callable I(t). Anything time dependent goes here,
        e.g. lambda t: 1 + 2*np.sin(t) for a modulated drive.
    sigma : white noise amplitude; the increment is sqrt(2*dt)*sigma*randn.
    Returns t, u, spikes, rate (Hz, 0 if fewer than two spikes).
    """
    rng = np.random.default_rng(seed)
    Ifun = I if callable(I) else (lambda t: I)

    nt = int(round(T / dt)) + 1
    t = np.linspace(0.0, T, nt)
    u = np.zeros(nt)
    u[0] = urest
    spikes = []

    for j in range(nt - 1):
        du = dt * (R * Ifun(t[j]) - (u[j] - urest)) / taum
        if sigma:
            du += np.sqrt(2 * dt) * sigma * rng.standard_normal()
        u[j + 1] = u[j] + du
        if u[j + 1] > uth:
            u[j + 1] = urest
            spikes.append(t[j + 1])

    spikes = np.array(spikes)
    rate = 1000 * (len(spikes) - 1) / (spikes[-1] - spikes[0]) if len(spikes) >= 2 else 0.0
    return t, u, spikes, rate


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = [
        ("constant $I=1.5$", 1.5, 0.0),
        ("sinusoidal $I=2\\sin t$", lambda t: 2 * np.sin(t), 0.0),
        ("constant $+$ noise", 0.95, 0.1),
    ]
    fig, ax = plt.subplots(len(cases), 1, figsize=(7, 7), sharex=True)
    for a, (lab, I, s) in zip(ax, cases):
        t, u, spk, rate = lif_sim(I, T=20.0, sigma=s, seed=0)
        a.plot(t, u, lw=1.2)
        a.plot(spk, np.full_like(spk, 1.0), "r.", ms=6)
        a.set_ylabel("$u$ (mV)")
        a.set_title(f"{lab}: {len(spk)} spikes, rate {rate:.1f} Hz", fontsize=10)
    ax[-1].set_xlabel("$t$ (ms)")
    fig.tight_layout()
    fig.savefig("lif_sim_demo.pdf")

    t, u, spk, rate = lif_sim()
    isi = np.diff(spk)
    pred = taum_isi = 1.0 * np.log(1.1 / (1.1 - 1.0))
    print(f"default case: mean ISI {isi.mean():.4f} ms, predicted {pred:.4f} ms")
