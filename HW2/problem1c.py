"""Problem 1c: Speed of the McKean traveling front vs. the passive apparent speed.

Front speed (from part b):   c(a) = sqrt(D) (1 - 2a) / sqrt(a(1 - a))
Passive apparent speed far from the injection site (from part a):   2 sqrt(D)

The front stalls at a = 1/2 and beats passive spread for a < a* = (2 - sqrt(2))/4.
"""

import numpy as np
import matplotlib.pyplot as plt


def front_speed(a, D=1.0):
    return np.sqrt(D) * (1 - 2 * a) / np.sqrt(a * (1 - a))


if __name__ == "__main__":
    D = 1.0
    c_passive = 2 * np.sqrt(D)
    a_star = (2 - np.sqrt(2)) / 4

    print(f"stall threshold a = 0.5,  c(0.5) = {front_speed(0.5, D):.4f}")
    print(f"crossing a* = {a_star:.4f},  c(a*) = {front_speed(a_star, D):.4f}  "
          f"(passive 2 sqrt(D) = {c_passive:.4f})")

    a = np.linspace(0.001, 0.999, 2000)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(a, front_speed(a, D), lw=2, label=r"active front $c(a)=\sqrt{D}\,(1-2a)/\sqrt{a(1-a)}$")
    ax.axhline(c_passive, color="k", ls="--", lw=1.2, label=r"passive apparent speed $2\sqrt{D}$")
    ax.axhline(0, color="gray", lw=0.8)
    ax.plot(a_star, c_passive, "ro", ms=7, label=rf"crossing $a^*=(2-\sqrt{{2}})/4\approx{a_star:.4f}$")
    ax.plot(0.5, 0, "ks", ms=7, label=r"stall $a=1/2$ ($c=0$)")
    ax.axvline(a_star, color="r", ls=":", lw=1)
    ax.set_ylim(-6, 6)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"threshold $a$ (dimensionless)")
    ax.set_ylabel(r"speed $c$ ($\lambda$ per $\tau_m$)")
    ax.set_title(r"McKean front speed vs. threshold, $D=1$")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("problem1c_speed_vs_a.png", dpi=150)
