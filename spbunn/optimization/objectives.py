from typing import Protocol, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class Func(Protocol):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ...

class LinearFunc(Func):
    def __init__(self, theta_0=1., theta_1=2., requires_grad=False) -> None:
        self.theta_0 = torch.tensor(
            theta_0,
            dtype=torch.float64,
            requires_grad=requires_grad
        )
        self.theta_1 = torch.tensor(
            theta_1,
            dtype=torch.float64,
            requires_grad=requires_grad,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.theta_0 + self.theta_1 * x


class DoubleSigmoidFunc(Func):
    def __init__(self, theta_0=1., theta_1=2., requires_grad=False) -> None:
        self.theta_0 = torch.tensor(
            theta_0,
            dtype=torch.float64,
            requires_grad=requires_grad,
        )
        self.theta_1 = torch.tensor(
            theta_1,
            dtype=torch.float64,
            requires_grad=requires_grad,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return 1. / (1. + torch.exp(-self.theta_0 / (1. + torch.exp(-self.theta_1*x))))


class DoubleReluFunc(Func):
    def __init__(self, theta_0=1., theta_1=2., requires_grad=False) -> None:
        self.theta_0 = torch.tensor(
            theta_0,
            dtype=torch.float64,
            requires_grad=requires_grad,
        )
        self.theta_1 = torch.tensor(
            theta_1,
            dtype=torch.float64,
            requires_grad=requires_grad,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.theta_0 * F.relu(self.theta_1 * x)


class SamplingFromFunctionDataset(Dataset):
    def __init__(
        self,
        f: Func,
        n: int,
        rng: Optional[np.random.Generator] = None,
        x_min: float = -10.0,
        x_max: float = 10.0,
    ) -> None:
        self.f = f
        self.n = n
        self._x = np.linspace(x_min, x_max, n)
        if rng is None:
            self._x
        else:
            self._x = rng.choice(self._x, size=n, replace=False)

    def __len__(self):
        return self.n
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx > self.n:
            raise KeyError(f"Index must be less than {self.n}") 

        # Generate from uniform(0, 1)
#        if self.rng is None:
#            x = self._x[idx]
#        else:
#            x = torch.tensor(self.rng.random(), dtype=torch.float64)  
        x = self._x[idx]
        y = self.f(x)
        return x, y

    
