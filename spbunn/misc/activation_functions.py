import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def relu(x):
    return np.where(x > 0, x, 0)


def relu_grad(x):
    return np.where(x > 0, 1, 0)


def leaky_relu(x, alpha=1e-1):
    return np.where(x > 0, x, alpha * x)


def leaky_relu_grad(x, alpha=1e-1):
    return np.where(x > 0, 1, alpha)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_grad(x):
    return np.exp(-x) / (1.0 + np.exp(-x)) ** 2


def tanh(x):
    return np.tanh(x)


def tanh_grad(x):
    return 1.0 - np.tanh(x) ** 2


def plot_func_and_grad(f):
    x = np.linspace(-5, 5, 200)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.plot(x, f(x), linewidth=4, label=r"$f(x)$")
    ax.plot(
        x, globals()[f"{f.__name__}_grad"](x), linewidth=4, alpha=0.5, label=r"$f'(x)$"
    )
    ax.grid()
    ax.legend()
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel(f"{f.__name__}", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{f.__name__}.svg")


def plot_fixed_point_iteration_histograms(f):
    n_sample = 1000
    n_iters = 100
    n_bins = 50
    x = np.random.uniform(-2, 2, size=n_sample)

    bins = None
    f_hists = np.zeros((n_bins, n_iters))
    for i in range(n_iters):
        if bins is None:
            f_hists[:, i], bins = np.histogram(x, bins=n_bins)
        else:
            f_hists[:, i], _ = np.histogram(x, bins=bins)

        x = f(x)

    f_hists /= n_sample

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    conts = ax.pcolormesh(
        np.arange(0.5, n_iters, 1),
        bins[:-1],
        f_hists,
        cmap=matplotlib.colormaps["ocean_r"],
    )
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Histogram", fontsize=12)
    plt.colorbar(conts)
    plt.tight_layout()
    plt.savefig(f"fp_iteration_hist_{f.__name__}.svg")


if __name__ == "__main__":
    for f in (tanh, sigmoid, relu, leaky_relu):
        plot_func_and_grad(f)
        plot_fixed_point_iteration_histograms(f)
