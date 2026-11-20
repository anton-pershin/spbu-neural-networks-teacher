"""Deterministic synthetic tasks used by the BPTT instability audit."""

from dataclasses import dataclass
from math import log

import torch

SUPPORTED_DTYPES = (torch.float32, torch.float64)


@dataclass(frozen=True)
class TaskBatch:
    """Input and target tensors for one synthetic audit batch."""

    inputs: torch.Tensor
    targets: torch.Tensor


def generate_copy_batch(
    *,
    symbol_count: int,
    memorize_length: int,
    gap: int,
    batch_size: int,
    seed: int,
    dtype: torch.dtype,
) -> TaskBatch:
    """Generate random symbols, a delimiter, a blank gap, and copy targets."""
    _positive_integer(symbol_count, "The symbol count")
    if symbol_count < 2:
        raise ValueError("The symbol count must be at least two")
    _positive_integer(memorize_length, "The memorized-symbol count")
    _non_negative_integer(gap, "The blank gap")
    _positive_integer(batch_size, "The batch size")
    _non_negative_integer(seed, "The data seed")
    _validate_dtype(dtype)

    generator = torch.Generator().manual_seed(seed)
    targets = torch.randint(
        0,
        symbol_count,
        (batch_size, memorize_length),
        generator=generator,
    )
    sequence_length = 2 * memorize_length + gap + 1
    indices = torch.full(
        (batch_size, sequence_length),
        symbol_count,
        dtype=torch.long,
    )
    indices[:, :memorize_length] = targets
    indices[:, memorize_length] = symbol_count + 1
    inputs = torch.nn.functional.one_hot(indices, num_classes=symbol_count + 2).to(
        torch.float32
    )
    return TaskBatch(inputs=inputs.to(dtype), targets=targets)


def generate_adding_batch(
    *,
    length: int,
    batch_size: int,
    seed: int,
    dtype: torch.dtype,
) -> TaskBatch:
    """Generate two marked random values and their exact sum per sequence."""
    _positive_integer(length, "The adding-sequence length")
    if length < 2:
        raise ValueError("The adding-sequence length must be at least two")
    _positive_integer(batch_size, "The batch size")
    _non_negative_integer(seed, "The data seed")
    _validate_dtype(dtype)

    generator = torch.Generator().manual_seed(seed)
    inputs = torch.zeros(batch_size, length, 2, dtype=torch.float32)
    inputs[:, :, 0] = torch.rand(batch_size, length, generator=generator)
    first = torch.randint(0, length, (batch_size,), generator=generator)
    offsets = torch.randint(1, length, (batch_size,), generator=generator)
    second = (first + offsets) % length
    rows = torch.arange(batch_size)
    inputs[rows, first, 1] = 1.0
    inputs[rows, second, 1] = 1.0
    targets = (inputs[:, :, 0] * inputs[:, :, 1]).sum(dim=1)
    return TaskBatch(inputs=inputs.to(dtype), targets=targets.to(dtype))


def copy_memoryless_loss(symbol_count: int) -> float:
    """Return cross-entropy for the optimal memoryless uniform predictor."""
    _positive_integer(symbol_count, "The symbol count")
    if symbol_count < 2:
        raise ValueError("The symbol count must be at least two")
    return log(symbol_count)


def adding_mean_baseline_mse() -> float:
    """Return the MSE of predicting the known target mean of one."""
    return 1.0 / 6.0


def _validate_dtype(dtype: torch.dtype) -> None:
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError("The audit dtype must be torch.float32 or torch.float64")


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _non_negative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
