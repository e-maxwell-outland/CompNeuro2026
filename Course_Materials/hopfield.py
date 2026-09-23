"""Hopfield networks: storage, retrieval, spurious states, and capacity.

Patterns xi^mu in {-1,+1}^N are stored in W = (1/N) sum_mu xi^mu (xi^mu)^T
and retrieved by iterating V -> sign(W V). Each stored pattern is a fixed
point, so activity started nearby is pulled onto it.

Three demonstrations:

1. One pattern, N = 10. Activity converges to near-perfect overlap in a
   few steps.
2. Several random patterns, growing P/N. As the load rises the network
   increasingly lands on spurious attractors -- mixtures of stored
   patterns that are themselves fixed points -- rather than on any
   pattern it was asked to store. The theoretical capacity is about
   0.138*N; well past that, blackout, and nothing is retrievable.
3. Denoising 32x32 images. Even where exact recovery fails on random
   patterns, structured patterns with low mutual overlap are recovered
   from a badly corrupted start.

Replaces Hopfield.m, which did not run: the three image patterns were
left as empty assignments (xis(:,1) = ;), a syntax error, and the final
denoising loop plotted a stale `dst` from the previous section. The
images here are generated rather than loaded, so nothing external is
needed; swap in your own bitmaps by replacing make_patterns.

Regression check: case 1 reaches overlap 1.0. Case 2 shows the retrieval
fraction falling as P/N rises past ~0.14. Case 3 recovers all three
images from 20 percent pixel flips.
"""

import numpy as np

__all__ = ["hebb_weights", "recall", "make_patterns", "retrieval_fraction"]


def hebb_weights(patterns):
    """W = (1/N) sum_mu xi xi^T, with the diagonal left in as in the original."""
    N = patterns.shape[0]
    return patterns @ patterns.T / N


def recall(W, V, nt=100):
    """Iterate V -> sign(W V) until fixed or nt steps. Returns V, n_steps."""
    for k in range(nt):
        Vn = np.sign(W @ V)
        Vn[Vn == 0] = 1
        if np.array_equal(Vn, V):
            return Vn, k
        V = Vn
    return V, nt


def make_patterns(N, P, rng):
    """P random +/-1 patterns of length N, as an (N, P) array."""
    return rng.choice([-1.0, 1.0], size=(N, P))


def retrieval_fraction(N, P, trials=40, rng=None):
    """Fraction of random starts that land exactly on a stored pattern."""
    rng = rng or np.random.default_rng(0)
    xis = make_patterns(N, P, rng)
    W = hebb_weights(xis)
    hits = 0
    for _ in range(trials):
        V, _ = recall(W, rng.choice([-1.0, 1.0], size=N))
        if np.abs(xis.T @ V).max() == N:
            hits += 1
    return hits / trials


def image_patterns(side=32):
    """Three structured 32x32 patterns: a cross, a ring, and diagonal bands."""
    g = np.arange(side)
    X, Y = np.meshgrid(g, g)
    c = (side - 1) / 2
    cross = np.where((np.abs(X - c) < 4) | (np.abs(Y - c) < 4), 1.0, -1.0)
    rad = np.hypot(X - c, Y - c)
    ring = np.where((rad > side * 0.25) & (rad < side * 0.4), 1.0, -1.0)
    bands = np.where(((X + Y) // 5) % 2 == 0, 1.0, -1.0)
    return np.stack([p.ravel() for p in (cross, ring, bands)], axis=1)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)

    # 1. single pattern
    xi = rng.choice([-1.0, 1.0], size=(10, 1))
    W = hebb_weights(xi)
    V, steps = recall(W, rng.choice([-1.0, 1.0], size=10))
    print(f"1. single pattern, N=10: overlap {abs(V @ xi[:,0])/10:.3f} in {steps} steps")

    # 2. capacity
    print("2. capacity, N = 200:")
    loads, fracs = [], []
    for P in [5, 10, 20, 28, 40, 60]:
        fr = retrieval_fraction(200, P, trials=30, rng=rng)
        loads.append(P / 200); fracs.append(fr)
        print(f"     P/N = {P/200:.3f}  retrieval fraction {fr:.2f}")

    # 3. denoising
    xis = image_patterns(32)
    W = hebb_weights(xis)
    flip = np.where(rng.random((1024, 3)) < 0.2, -1.0, 1.0)
    noisy = flip * xis
    fig, ax = plt.subplots(3, 3, figsize=(7.5, 7.5))
    print("3. denoising 32x32 images from 20% pixel flips:")
    for j in range(3):
        V, _ = recall(W, noisy[:, j].copy())
        ov = abs(V @ xis[:, j]) / 1024
        print(f"     pattern {j+1}: overlap after recall {ov:.3f}")
        for a, img, lab in zip(ax[j], [xis[:, j], noisy[:, j], V],
                               ["stored", "corrupted", "recalled"]):
            a.imshow(img.reshape(32, 32), cmap="gray"); a.set_xticks([]); a.set_yticks([])
            if j == 0: a.set_title(lab)
    fig.tight_layout(); fig.savefig("hopfield_denoise.pdf")

    fig, a = plt.subplots(figsize=(5.6, 4.2))
    a.plot(loads, fracs, "ko-", lw=1.5)
    a.axvline(0.138, ls="--", color="r", label="theoretical capacity 0.138")
    a.set_xlabel("load $P/N$"); a.set_ylabel("retrieval fraction"); a.legend()
    fig.tight_layout(); fig.savefig("hopfield_capacity.pdf")
