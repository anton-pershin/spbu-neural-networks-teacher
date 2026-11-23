"""Boundary-relative RNN and LSTM gradient audits for trained copy models."""

from dataclasses import dataclass

import torch

from spbunn.rnn.audit import DtypeName
from spbunn.rnn.copy_training import CopyTaskSettings, copy_cross_entropy
from spbunn.rnn.models import CopyLSTMModel, CopyRNNModel
from spbunn.rnn.tasks import TaskBatch


@dataclass(frozen=True)
class RNNGradientAudit:
    """Full hidden-state adjoints in increasing-lag order."""

    gap: int
    dtype: DtypeName
    loss: float
    relative_state_gradients: tuple[float, ...]
    valid_sequence_count: int
    zero_state_gradient_count: int
    nonfinite_state_gradient_count: int


@dataclass(frozen=True)
class LSTMGradientAudit:
    """Full and direct cell-path adjoints plus forget-gate diagnostics."""

    gap: int
    dtype: DtypeName
    loss: float
    relative_cell_gradients: tuple[float, ...]
    direct_cell_path_gradients: tuple[float, ...]
    mean_gate_products: tuple[float, ...]
    mean_forget_gates: tuple[float, ...]
    valid_sequence_count: int
    zero_cell_gradient_count: int
    nonfinite_cell_gradient_count: int


def audit_rnn_gradients(
    model: CopyRNNModel, batch: TaskBatch, task: CopyTaskSettings
) -> RNNGradientAudit:
    """Measure every pre-reproduction hidden adjoint with one backward pass."""
    _validate_batch(model, batch, task)
    model.zero_grad(set_to_none=True)
    unroll = model.unroll(batch.inputs, retain_states=True)
    loss = copy_cross_entropy(unroll.logits, batch.targets)
    loss.backward()
    norms = _state_gradient_norms(unroll.states, "hidden")
    reference = task.memorize_length + task.gap
    relative, valid_count = _relative_norms(norms, reference)
    result = RNNGradientAudit(
        gap=task.gap,
        dtype=_dtype_name(batch.inputs.dtype),
        loss=float(loss.detach()),
        relative_state_gradients=relative,
        valid_sequence_count=valid_count,
        zero_state_gradient_count=int((norms == 0).sum()),
        nonfinite_state_gradient_count=int((~torch.isfinite(norms)).sum()),
    )
    model.zero_grad(set_to_none=True)
    return result


def audit_lstm_gradients(
    model: CopyLSTMModel, batch: TaskBatch, task: CopyTaskSettings
) -> LSTMGradientAudit:
    """Measure full and direct cell-path adjoints with one backward pass."""
    _validate_batch(model, batch, task)
    model.zero_grad(set_to_none=True)
    unroll = model.unroll(batch.inputs, retain_states=True)
    loss = copy_cross_entropy(unroll.logits, batch.targets)
    loss.backward()
    norms = _state_gradient_norms(unroll.cell_states, "cell")
    reference = task.memorize_length + task.gap
    relative, valid_count = _relative_norms(norms, reference)
    reference_gradients = unroll.cell_states[reference].grad
    if reference_gradients is None:
        raise RuntimeError("Missing retained reference cell gradient")
    reference_gradients = reference_gradients.detach().to(torch.float64)
    reference_norms = reference_gradients.norm(dim=-1)
    valid = torch.isfinite(reference_norms) & (reference_norms > 0)
    direct = _direct_cell_path_norms(
        reference_gradients=reference_gradients,
        forget_gates=unroll.forget_gates,
        reference=reference,
        valid=valid,
    )
    mean_forget = (
        torch.stack(unroll.forget_gates, dim=0)
        .detach()
        .to(torch.float64)
        .mean(dim=(1, 2))
    )
    mean_products = _mean_gate_products(mean_forget, reference)
    result = LSTMGradientAudit(
        gap=task.gap,
        dtype=_dtype_name(batch.inputs.dtype),
        loss=float(loss.detach()),
        relative_cell_gradients=relative,
        direct_cell_path_gradients=direct,
        mean_gate_products=mean_products,
        mean_forget_gates=tuple(float(value) for value in mean_forget),
        valid_sequence_count=valid_count,
        zero_cell_gradient_count=int((norms == 0).sum()),
        nonfinite_cell_gradient_count=int((~torch.isfinite(norms)).sum()),
    )
    model.zero_grad(set_to_none=True)
    return result


def _direct_cell_path_norms(
    *,
    reference_gradients: torch.Tensor,
    forget_gates: tuple[torch.Tensor, ...],
    reference: int,
    valid: torch.Tensor,
) -> tuple[float, ...]:
    """Calculate cumulative vector-valued gate products in linear time."""
    batch_size = reference_gradients.shape[0]
    ratios = torch.full((batch_size, reference + 1), float("nan"), dtype=torch.float64)
    reference_norms = reference_gradients.norm(dim=-1)
    product = torch.ones_like(reference_gradients)
    for time_index in range(reference, -1, -1):
        if time_index < reference:
            product = product * forget_gates[time_index + 1].detach().to(torch.float64)
        direct_norms = (reference_gradients * product).norm(dim=-1)
        ratios[valid, time_index] = direct_norms[valid] / reference_norms[valid]
    mean_ratios = ratios[valid].mean(dim=0) if bool(valid.any()) else ratios[0]
    return tuple(float(value) for value in mean_ratios.flip(0))


def _mean_gate_products(mean_forget: torch.Tensor, reference: int) -> tuple[float, ...]:
    """Return products of scalar mean gates in increasing-lag order."""
    products = torch.ones(reference + 1, dtype=torch.float64)
    product = torch.ones((), dtype=torch.float64)
    for time_index in range(reference - 1, -1, -1):
        product = product * mean_forget[time_index + 1]
        products[time_index] = product
    return tuple(float(value) for value in products.flip(0))


def _state_gradient_norms(
    states: tuple[torch.Tensor, ...], state_name: str
) -> torch.Tensor:
    gradients = []
    for index, state in enumerate(states):
        if state.grad is None:
            raise RuntimeError(
                f"Missing retained {state_name}-state gradient at {index}"
            )
        gradients.append(state.grad.detach().to(torch.float64).norm(dim=-1))
    return torch.stack(gradients, dim=1)


def _relative_norms(
    norms: torch.Tensor, reference: int
) -> tuple[tuple[float, ...], int]:
    reference_norms = norms[:, reference]
    valid = torch.isfinite(reference_norms) & (reference_norms > 0)
    ratios = torch.full_like(norms[:, : reference + 1], float("nan"))
    if not bool(valid.any()):
        return tuple([float("nan")] * (reference + 1)), 0
    ratios[valid] = norms[valid, : reference + 1] / reference_norms[valid].unsqueeze(1)
    mean_ratios = ratios[valid].mean(dim=0).flip(0)
    return tuple(float(value) for value in mean_ratios), int(valid.sum())


def _validate_batch(
    model: CopyRNNModel | CopyLSTMModel,
    batch: TaskBatch,
    task: CopyTaskSettings,
) -> None:
    if batch.inputs.ndim != 3 or batch.targets.ndim != 2:
        raise ValueError("Copy audit inputs or targets have invalid dimensions")
    expected_time = 2 * task.memorize_length + task.gap + 1
    if batch.inputs.shape[1] != expected_time:
        raise ValueError("The copy audit gap does not match the batch time dimension")
    if batch.targets.shape != (batch.inputs.shape[0], task.memorize_length):
        raise ValueError("The copy audit targets have an incompatible shape")
    model_dtype = model.recurrent.weight_ih_l0.dtype
    if batch.inputs.dtype != model_dtype:
        raise ValueError("The copy audit batch dtype must match the model dtype")


def _dtype_name(dtype: torch.dtype) -> DtypeName:
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float64:
        return "float64"
    raise ValueError("The copy audit dtype must be torch.float32 or torch.float64")
