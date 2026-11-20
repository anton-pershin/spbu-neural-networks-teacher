"""Vanilla-RNN construction and retained-state BPTT gradient audits."""

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from spbunn.rnn.tasks import SUPPORTED_DTYPES, TaskBatch

TaskKind = Literal["copy", "adding"]
Initialization = Literal["default", "orthogonal", "xavier_tanh"]
DtypeName = Literal["float32", "float64"]


@dataclass(frozen=True)
class GradientAudit:
    """Numerical results from one forward and backward recurrent audit."""

    task: TaskKind
    initialization: Initialization
    seed: int
    task_length: int
    dtype: DtypeName
    batch_size: int
    hidden_size: int
    loss: float
    maximum_hidden_state: float
    parameter_gradient_norm: float
    relative_state_gradients: tuple[float, ...]
    valid_sequence_count: int
    zero_state_gradient_count: int
    nonfinite_state_gradient_count: int
    forward_finite: bool
    backward_finite: bool


class RecurrentAuditModel(nn.Module):
    """One explicitly unrolled RNNCell and a linear readout."""

    def __init__(self, input_size: int, output_size: int, hidden_size: int) -> None:
        super().__init__()
        self.cell = nn.RNNCell(input_size, hidden_size, nonlinearity="tanh")
        self.readout = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Return every separate hidden state and all readout logits."""
        hidden = inputs.new_zeros(inputs.shape[0], self.hidden_size)
        states = []
        for time_index in range(inputs.shape[1]):
            hidden = self.cell(inputs[:, time_index], hidden)
            if hidden.requires_grad:
                hidden.retain_grad()
            states.append(hidden)
        logits = self.readout(torch.stack(states, dim=1))
        return tuple(states), logits


def build_recurrent_model(
    *,
    input_size: int,
    output_size: int,
    hidden_size: int,
    initialization: Initialization,
    seed: int,
    dtype: torch.dtype,
) -> RecurrentAuditModel:
    """Construct a seeded float32 model and then cast it to the audit dtype."""
    if initialization not in {"default", "orthogonal", "xavier_tanh"}:
        raise ValueError(f"Unsupported recurrent initialization: {initialization}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("The model seed must be a non-negative integer")
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError("The audit dtype must be torch.float32 or torch.float64")

    # Model construction must not replace the caller's global random state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = RecurrentAuditModel(input_size, output_size, hidden_size)
        if initialization == "orthogonal":
            nn.init.orthogonal_(model.cell.weight_hh)
        elif initialization == "xavier_tanh":
            gain = nn.init.calculate_gain("tanh")
            nn.init.xavier_normal_(model.cell.weight_ih, gain=gain)
            nn.init.xavier_normal_(model.cell.weight_hh, gain=gain)
            nn.init.zeros_(model.cell.bias_ih)
            nn.init.zeros_(model.cell.bias_hh)
    model.to(dtype=dtype)
    return model


def audit_gradients(
    *,
    task: TaskKind,
    batch: TaskBatch,
    initialization: Initialization,
    seed: int,
    task_length: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> GradientAudit:
    """Measure retained hidden-state adjoints for one synthetic task batch."""
    if task not in {"copy", "adding"}:
        raise ValueError(f"Unsupported audit task: {task}")
    if batch.inputs.ndim != 3:
        raise ValueError("Audit inputs must have batch, time, and feature dimensions")
    if batch.inputs.dtype != dtype or (
        task == "adding" and batch.targets.dtype != dtype
    ):
        raise ValueError("The generated batch dtype must match the audit dtype")
    if (
        isinstance(task_length, bool)
        or not isinstance(task_length, int)
        or task_length < 0
    ):
        raise ValueError("The task length must be a non-negative integer")
    if task == "copy":
        if batch.targets.ndim != 2:
            raise ValueError("Copy targets must have batch and reproduction dimensions")
        expected_time = 2 * batch.targets.shape[1] + task_length + 1
    else:
        if batch.targets.ndim != 1:
            raise ValueError("Adding targets must have only a batch dimension")
        expected_time = task_length
    if batch.inputs.shape[1] != expected_time:
        raise ValueError(
            "The task length metadata is inconsistent with the batch time dimension: "
            f"expected {expected_time}, got {batch.inputs.shape[1]}"
        )

    model = build_recurrent_model(
        input_size=batch.inputs.shape[-1],
        output_size=batch.inputs.shape[-1] if task == "copy" else 1,
        hidden_size=hidden_size,
        initialization=initialization,
        seed=seed,
        dtype=dtype,
    )
    states, logits = model(batch.inputs)
    if task == "copy":
        target_length = batch.targets.shape[1]
        loss = nn.functional.cross_entropy(
            logits[:, -target_length:].reshape(-1, logits.shape[-1]),
            batch.targets.reshape(-1),
        )
    else:
        loss = nn.functional.mse_loss(logits[:, -1, 0], batch.targets)

    hidden_finite = all(bool(torch.isfinite(state.detach()).all()) for state in states)
    maximum_hidden = 0.0
    for state in states:
        value = float(state.detach().abs().max())
        maximum_hidden = (
            float("nan") if math.isnan(value) else max(maximum_hidden, value)
        )
    loss.backward()
    backward_norms = _state_gradient_norms(states)
    parameter_norm = _parameter_gradient_norm(model)
    ratios, valid_count = _relative_norms(backward_norms)
    zero_count = int((backward_norms == 0).sum())
    nonfinite_count = int((~torch.isfinite(backward_norms)).sum())
    backward_finite = nonfinite_count == 0 and math.isfinite(parameter_norm)

    return GradientAudit(
        task=task,
        initialization=initialization,
        seed=seed,
        task_length=task_length,
        dtype=_dtype_name(dtype),
        batch_size=batch.inputs.shape[0],
        hidden_size=hidden_size,
        loss=float(loss.detach()),
        maximum_hidden_state=maximum_hidden,
        parameter_gradient_norm=parameter_norm,
        relative_state_gradients=tuple(float(value) for value in ratios),
        valid_sequence_count=valid_count,
        zero_state_gradient_count=zero_count,
        nonfinite_state_gradient_count=nonfinite_count,
        forward_finite=hidden_finite and math.isfinite(float(loss.detach())),
        backward_finite=backward_finite,
    )


def _state_gradient_norms(states: tuple[torch.Tensor, ...]) -> torch.Tensor:
    gradients = []
    for index, state in enumerate(states):
        if state.grad is None:
            raise RuntimeError(
                f"Missing retained hidden-state gradient at index {index}"
            )
        gradients.append(state.grad.detach().to(torch.float64).norm(dim=-1))
    return torch.stack(gradients, dim=1)


def _relative_norms(norms: torch.Tensor) -> tuple[torch.Tensor, int]:
    final_norms = norms[:, -1]
    valid = torch.isfinite(final_norms) & (final_norms > 0)
    ratios = torch.full_like(norms, float("nan"))
    if not bool(valid.any()):
        return torch.full_like(norms[0], float("nan")), 0
    ratios[valid] = norms[valid] / final_norms[valid].unsqueeze(1)
    mean_ratios = ratios[valid].mean(dim=0)
    return mean_ratios, int(valid.sum())


def _parameter_gradient_norm(model: RecurrentAuditModel) -> float:
    squared_sum = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            gradient = parameter.grad.detach().to(torch.float64)
            squared_sum += float((gradient * gradient).sum())
    return math.sqrt(squared_sum)


def _dtype_name(dtype: torch.dtype) -> DtypeName:
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float64:
        return "float64"
    raise ValueError("The audit dtype must be torch.float32 or torch.float64")
