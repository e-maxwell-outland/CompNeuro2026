"""Backpropagation in small perceptrons, from one unit up to XOR.

Three examples, in increasing order of what a network can represent:

1. One unit, one input: z = f(w x + h). Trained so x=0 -> 0.1 and
   x=1 -> 0.9. A single sigmoid unit is monotone in x, so this is the
   most it can do.
2. One input, two hidden units, one output. Trained so x=0 -> 0.1,
   x=1 -> 0.9, x=2 -> 0.1. Nonmonotone, so the hidden layer is doing the
   work: no single unit can rise then fall.
3. Two inputs, two hidden units, one output: XOR. The classic case that
   cannot be done without a hidden layer at all, since XOR is not
   linearly separable.

Error is squared difference from target, E = (z - z_target)^2, and the
reported quantity is the largest error across all training conditions.
Weight updates follow the chain rule: delta at the output is
2(z - target) f'(net), and delta at a hidden unit is that times the
outgoing weight times f'(net) of the hidden unit.

Replaces Backprop_simple.ipynb, trainxor.py, and trainxorloop.py, which
between them wrote the same update rule out by hand roughly forty times.

Two bugs fixed by construction. Both XOR scripts had, in the x1=x2=1
block, two consecutive updates assigning to J1, the second of which was
meant for J2, so the second output weight never received its fourth
update. And both scripts' verification blocks computed y1 = f(x1+h1)
rather than the actual forward pass f(w11*x1+w12*x2+h1), so the printed
check did not test the trained network. Writing the updates as matrix
operations makes the first class of bug impossible and lets the same
forward() serve both training and checking.

Regression check: all three cases should drive the max error below 1e-4.
XOR with two hidden units stalls in a local minimum for roughly 2 seeds
in 12 (the network settles with two of the four cases at 0.5 and cannot
escape). That is a real property of the minimal architecture, not a bug,
and is worth demonstrating: three hidden units converged on 12 of 12
seeds in the same test. The demo uses seed 1, which stalls at two
hidden units and succeeds at three.
"""

import numpy as np

__all__ = ["f", "fp", "forward", "train_perceptron", "train_mlp"]


def f(x):
    return 1 / (1 + np.exp(-x))


def fp(x):
    return np.exp(-x) / (1 + np.exp(-x)) ** 2


def forward(X, W1, b1, W2, b2):
    """One hidden layer. X is (n_samples, n_in). Returns z, y, nets."""
    net1 = X @ W1 + b1
    y = f(net1)
    net2 = y @ W2 + b2
    return f(net2), y, net1, net2


def train_perceptron(X, T, rate=1.0, n_iter=10000, tol=1e-4, seed=0):
    """Single sigmoid unit z = f(w.x + h). Returns w, h, error history."""
    rng = np.random.default_rng(seed)
    X = np.atleast_2d(np.asarray(X, float).reshape(len(T), -1))
    T = np.asarray(T, float)
    w = rng.uniform(-1, 1, X.shape[1])
    h = rng.uniform(-1, 1)
    hist = []
    for i in range(n_iter):
        net = X @ w + h
        z = f(net)
        hist.append(np.max((z - T) ** 2))
        d = 2 * (z - T) * fp(net)
        w -= rate * (d @ X) / len(T)
        h -= rate * d.mean()
        if hist[-1] < tol:
            break
    return w, h, np.array(hist)


def train_mlp(X, T, n_hidden=2, rate=1.0, n_iter=100000, tol=1e-4, seed=0):
    """One hidden layer. Returns weights, biases, error history."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float).reshape(len(T), -1)
    T = np.asarray(T, float)
    n_in = X.shape[1]
    W1 = rng.uniform(-1, 1, (n_in, n_hidden))
    b1 = rng.uniform(-1, 1, n_hidden)
    W2 = rng.uniform(-1, 1, n_hidden)
    b2 = rng.uniform(-1, 1)
    hist = []
    for i in range(n_iter):
        z, y, net1, net2 = forward(X, W1, b1, W2, b2)
        hist.append(np.max((z - T) ** 2))
        if hist[-1] < tol:
            break
        d2 = 2 * (z - T) * fp(net2)                    # output delta
        d1 = np.outer(d2, W2) * fp(net1)               # hidden deltas
        W2 -= rate * (y.T @ d2) / len(T)
        b2 -= rate * d2.mean()
        W1 -= rate * (X.T @ d1) / len(T)
        b1 -= rate * d1.mean(axis=0)
    return (W1, b1, W2, b2), np.array(hist)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))

    # 1. single unit, monotone
    w, h, hist = train_perceptron([[0], [1]], [0.1, 0.9], rate=2.0)
    z = f(np.array([[0], [1]]) @ w + h)
    print(f"1. single unit   : {len(hist):6d} iters, outputs {np.round(z,4)}")
    ax[0].semilogy(hist, lw=2); ax[0].set_title("one unit, one input")

    # 2. hidden layer, nonmonotone
    (W1, b1, W2, b2), hist = train_mlp([[0], [1], [2]], [0.1, 0.9, 0.1],
                                       n_hidden=2, rate=2.0, seed=3)
    z, *_ = forward(np.array([[0], [1], [2]]), W1, b1, W2, b2)
    print(f"2. nonmonotone   : {len(hist):6d} iters, outputs {np.round(z,4)}")
    ax[1].semilogy(hist, lw=2); ax[1].set_title("two hidden, nonmonotone")

    # 3. XOR
    Xx = [[0, 0], [1, 0], [0, 1], [1, 1]]
    Tx = [0.1, 0.9, 0.9, 0.1]
    for nh, style in [(2, "r-"), (3, "k-")]:
        (W1, b1, W2, b2), hist = train_mlp(Xx, Tx, n_hidden=nh, rate=5.0, seed=1)
        z, *_ = forward(np.array(Xx, float), W1, b1, W2, b2)
        tag = "converged" if hist[-1] < 1e-4 else "STALLED"
        print(f"3. XOR, {nh} hidden: {len(hist):6d} iters, outputs {np.round(z,4)}  {tag}")
        ax[2].semilogy(hist, style, lw=2, label=f"{nh} hidden ({tag})")
    ax[2].set_title("XOR (seed 1)"); ax[2].legend(fontsize=8)

    for a in ax:
        a.set_xlabel("iteration"); a.set_ylabel("max error")
    fig.tight_layout(); fig.savefig("backprop_demo.pdf")
