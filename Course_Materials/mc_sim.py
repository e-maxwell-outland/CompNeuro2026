"""Morris-Lecar model of a barnacle muscle fiber.

    C u' = I - gL(u - VL) - g2 w (u - V2) - g1 m0(u)(u - V1)
      w' = (w0(u) - w) / tau(u)

with instantaneous calcium activation m0 and slow potassium activation w.
Two variables like FitzHugh-Nagumo, but the nonlinearity comes from real
conductances rather than a fitted cubic, so the parameters mean something.

The u3, u4 pair sets where and how sharply potassium activates, and it is
what selects the bifurcation: the values here give a Hopf onset, while the
commented alternatives (u3 = 12, u4 = 17.4) give a SNIC, with a firing rate
that rises continuously from zero. Worth running both.

Fixed from the original: it saved its figures as fn_model_tser.png and
fn_model_pplane.png, copied from fn_mod.py, so running it after the
FitzHugh-Nagumo script silently overwrote those figures.

Regression check: with the defaults the model fires repetitively; u should
oscillate roughly between -30 and +30 mV.
"""

import numpy as np

__all__ = ["mc_sim"]

# defaults: Hopf regime. For the SNIC regime use u3 = 12, u4 = 17.4.
PARAMS = dict(tauw=25.0, g1=4.4, g2=8.0, gL=2.0, V1=120.0, V2=-84.0,
              VL=-60.0, u1=-1.2, u2=18.0, u3=2.0, u4=30.0, C=20.0)


def mc_sim(I=90.0, T=1000.0, dt=0.01, u0=-30.0, w0init=0.02, **kw):
    """Euler integration of Morris-Lecar. Returns t, u, w."""
    p = {**PARAMS, **kw}
    m0 = lambda u: (1 + np.tanh((u - p["u1"]) / p["u2"])) / 2
    w0 = lambda u: (1 + np.tanh((u - p["u3"]) / p["u4"])) / 2
    tau = lambda u: p["tauw"] / np.cosh((u - p["u3"]) / (2 * p["u4"]))

    nt = int(round(T / dt)) + 1
    t = np.linspace(0, T, nt)
    u = np.zeros(nt); w = np.zeros(nt)
    u[0], w[0] = u0, w0init
    for j in range(nt - 1):
        u[j + 1] = u[j] + dt * (I - p["gL"] * (u[j] - p["VL"])
                                - p["g2"] * w[j] * (u[j] - p["V2"])
                                - p["g1"] * m0(u[j]) * (u[j] - p["V1"])) / p["C"]
        w[j + 1] = w[j] + dt * (w0(u[j]) - w[j]) / tau(u[j])
    return t, u, w


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, u, w = mc_sim()
    print(f"u range: {u[-20000:].min():.1f} to {u[-20000:].max():.1f} mV")

    fig, a = plt.subplots(figsize=(7, 3.6))
    a.plot(t, u, lw=1.5, label="$u$ (mV)")
    a.plot(t, 100 * w, lw=1.5, label="$100\\,w$")
    a.set_xlabel("$t$ (ms)"); a.set_ylabel("voltage / recovery"); a.legend()
    fig.tight_layout(); fig.savefig("mc_timeseries.pdf")

    fig, a = plt.subplots(figsize=(5.4, 4.4))
    a.plot(u, w, lw=1.2)
    a.set_xlabel("voltage $u$ (mV)"); a.set_ylabel("recovery $w$")
    fig.tight_layout(); fig.savefig("mc_phaseplane.pdf")
