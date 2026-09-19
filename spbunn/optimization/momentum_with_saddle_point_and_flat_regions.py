import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define a function with saddle point and flat regions
def saddle_function(x, y):
    """Function with saddle point at (0,0) and flat regions"""
    return (x**2 - y**2) * torch.exp(-0.5*(x**2 + y**2)) + 0.2 * torch.sin(5*x) + 0.2 * torch.sin(5*y)

# Create grid for visualization
x = torch.linspace(-3.5, 3.5, 100)
y = torch.linspace(-3.5, 3.5, 100)
X, Y = torch.meshgrid(x, y, indexing='xy')
Z = saddle_function(X, Y)

# Initialize parameters
def initialize_params():
    #return torch.tensor([1.2, 1.2], requires_grad=True)
    return torch.tensor([-2.0, 1.3], requires_grad=True)

# Training function
def train_with_optimizer(optimizer_class, optimizer_params, num_epochs=100):
    params = initialize_params()
    optimizer = optimizer_class([params], **optimizer_params)
    
    trajectory = []
    losses = []
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        x, y = params
        loss = saddle_function(x, y)
        loss.backward()
        optimizer.step()
        
        trajectory.append(params.detach().clone().numpy())
        losses.append(loss.item())
    
    return np.array(trajectory), losses

# Compare SGD with and without momentum
num_epochs = 200

# SGD without momentum
sgd_trajectory, sgd_losses = train_with_optimizer(
    optim.SGD, {'lr': 8*1e-2, 'momentum': 0.0}, num_epochs
)

# SGD with momentum
momentum_trajectory, momentum_losses = train_with_optimizer(
    optim.SGD, {'lr': 8*1e-2, 'momentum': 0.9}, num_epochs
)

# Create visualization
fig = plt.figure(figsize=(18, 12))

# 3D surface plot
ax1 = fig.add_subplot(231, projection='3d')
surf = ax1.plot_surface(X.numpy(), Y.numpy(), Z.numpy(), 
                       cmap=cm.coolwarm, alpha=0.6, linewidth=0, antialiased=True)
ax1.scatter(sgd_trajectory[:, 0], sgd_trajectory[:, 1], 
           saddle_function(torch.tensor(sgd_trajectory[:, 0]), torch.tensor(sgd_trajectory[:, 1])).numpy(),
           color='red', s=20, label='SGD', alpha=0.8)
ax1.scatter(momentum_trajectory[:, 0], momentum_trajectory[:, 1], 
           saddle_function(torch.tensor(momentum_trajectory[:, 0]), torch.tensor(momentum_trajectory[:, 1])).numpy(),
           color='blue', s=20, label='SGD+Momentum', alpha=0.8)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Loss')
ax1.set_title('3D Optimization Trajectory')
ax1.legend()

# Contour plot
ax2 = fig.add_subplot(232)
contour = ax2.contour(X.numpy(), Y.numpy(), Z.numpy(), 20, cmap='coolwarm')
ax2.plot(sgd_trajectory[:, 0], sgd_trajectory[:, 1], 'ro-', 
        markersize=3, linewidth=1.5, label='SGD', alpha=0.7)
ax2.plot(momentum_trajectory[:, 0], momentum_trajectory[:, 1], 'bo-', 
        markersize=3, linewidth=1.5, label='SGD+Momentum', alpha=0.7)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Contour Plot with Trajectories')
ax2.legend()
plt.colorbar(contour, ax=ax2)

# Loss comparison
ax3 = fig.add_subplot(233)
ax3.plot(sgd_losses, 'r-', label='SGD', alpha=0.8)
ax3.plot(momentum_losses, 'b-', label='SGD+Momentum', alpha=0.8)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss')
ax3.set_title('Loss Convergence')
ax3.legend()
ax3.set_yscale('log')

# X coordinate over time
ax4 = fig.add_subplot(234)
ax4.plot(sgd_trajectory[:, 0], 'r-', label='SGD X', alpha=0.8)
ax4.plot(momentum_trajectory[:, 0], 'b-', label='Momentum X', alpha=0.8)
ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax4.set_xlabel('Epoch')
ax4.set_ylabel('X Value')
ax4.set_title('X Coordinate Convergence')
ax4.legend()

# Y coordinate over time
ax5 = fig.add_subplot(235)
ax5.plot(sgd_trajectory[:, 1], 'r-', label='SGD Y', alpha=0.8)
ax5.plot(momentum_trajectory[:, 1], 'b-', label='Momentum Y', alpha=0.8)
ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax5.set_xlabel('Epoch')
ax5.set_ylabel('Y Value')
ax5.set_title('Y Coordinate Convergence')
ax5.legend()

# Final positions
ax6 = fig.add_subplot(236)
final_positions = [
    ('SGD', sgd_trajectory[-1], 'red'),
    ('Momentum', momentum_trajectory[-1], 'blue')
]

for name, pos, color in final_positions:
    ax6.plot(pos[0], pos[1], 'o', markersize=10, color=color, label=name)
    ax6.text(pos[0] + 0.05, pos[1] + 0.05, f'{name}\n({pos[0]:.3f}, {pos[1]:.3f})')

contour = ax6.contour(X.numpy(), Y.numpy(), Z.numpy(), 10, cmap='coolwarm', alpha=0.6)
ax6.set_xlabel('X')
ax6.set_ylabel('Y')
ax6.set_title('Final Positions')
ax6.legend()

plt.tight_layout()
plt.show()

# Print final results
print("Final Results:")
print(f"SGD: Position = ({sgd_trajectory[-1, 0]:.6f}, {sgd_trajectory[-1, 1]:.6f}), "
      f"Loss = {sgd_losses[-1]:.6f}")
print(f"SGD+Momentum: Position = ({momentum_trajectory[-1, 0]:.6f}, {momentum_trajectory[-1, 1]:.6f}), "
      f"Loss = {momentum_losses[-1]:.6f}")
