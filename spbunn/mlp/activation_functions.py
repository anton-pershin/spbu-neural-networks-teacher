import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def relu(x):
    return np.where(x > 0, x, 0)

def sigmoid(x):
    return 1./(1. + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def plot_fixed_point_iteration_histograms(f):
    n_sample = 1000
    n_iters = 10
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

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    conts = ax.pcolormesh(np.arange(0.5, n_iters, 1), bins[:-1], f_hists, cmap=matplotlib.colormaps["ocean_r"])
    plt.colorbar(conts)
    plt.savefig(f"fp_iteration_hist_{f.__name__}.svg")


if __name__ == "__main__":
    plot_fixed_point_iteration_histograms(tanh)
    plot_fixed_point_iteration_histograms(sigmoid)
    plot_fixed_point_iteration_histograms(relu)
