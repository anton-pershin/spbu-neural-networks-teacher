import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from spbunn.mlp.simple_relu_nn import DeepReLUModel
from spbunn.resnet.residual_connections import DeepReLUModelWithResidualConnections


def plot_gradient_wrt_inputs(network, filename):
    x = torch.linspace(-1, 1, 256, requires_grad=True, dtype=torch.float32)
    output = network(x.unsqueeze(dim=1)).sum() 
    print(output)
    output.backward()
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.plot(x.detach().numpy(), x.grad.detach().numpy(), "o--", markersize=4)
    ax.grid()
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel(r"$dh / dx$", fontsize=12)
    fig.tight_layout()
    fig.savefig(filename)


def balduzzi_analysis():
    plain_network = DeepReLUModel(hidden_layer_dim=200, n_layers=24)
    res_network = DeepReLUModelWithResidualConnections(hidden_layer_dim=200, n_layers=24)
    plot_gradient_wrt_inputs(plain_network, "balduzzi_analysis_plain_network.svg")
    plot_gradient_wrt_inputs(res_network, "balduzzi_analysis_res_network.svg")

if __name__ == "__main__":
    balduzzi_analysis()

