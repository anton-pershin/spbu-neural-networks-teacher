"""Synthetic task structure, determinism, and validation checks."""

import math
from typing import Any

import pytest
import torch

from spbunn.rnn.tasks import (
    adding_mean_baseline_mse,
    copy_memoryless_loss,
    generate_adding_batch,
    generate_copy_batch,
)


def test_copy_batch_has_exact_symbols_delimiter_blanks_and_targets() -> None:
    batch = generate_copy_batch(
        symbol_count=8,
        memorize_length=3,
        gap=2,
        batch_size=4,
        seed=19,
        dtype=torch.float32,
    )

    assert batch.inputs.shape == (4, 9, 10)
    assert batch.targets.shape == (4, 3)
    indices = batch.inputs.argmax(dim=-1)
    assert torch.equal(indices[:, :3], batch.targets)
    assert torch.all(indices[:, 3] == 9)
    assert torch.all(indices[:, 4:6] == 8)
    assert torch.all(indices[:, 6:] == 8)
    assert torch.all(batch.targets >= 0)
    assert torch.all(batch.targets < 8)
    assert torch.equal(batch.inputs.sum(dim=-1), torch.ones(4, 9))


def test_adding_batch_has_two_distinct_markers_and_exact_target() -> None:
    batch = generate_adding_batch(
        length=12, batch_size=16, seed=23, dtype=torch.float32
    )

    assert batch.inputs.shape == (16, 12, 2)
    assert batch.targets.shape == (16,)
    assert torch.all(batch.inputs[:, :, 0] >= 0)
    assert torch.all(batch.inputs[:, :, 0] <= 1)
    assert torch.equal(batch.inputs[:, :, 1].sum(dim=1), torch.full((16,), 2.0))
    for inputs, target in zip(batch.inputs, batch.targets, strict=True):
        positions = inputs[:, 1].nonzero().flatten()
        assert len(positions) == 2
        assert positions[0] != positions[1]
        expected = inputs[positions, 0].sum()
        assert target == expected


def test_batches_are_deterministic_and_matched_across_dtypes() -> None:
    copy32 = generate_copy_batch(
        symbol_count=8,
        memorize_length=10,
        gap=20,
        batch_size=8,
        seed=99,
        dtype=torch.float32,
    )
    copy32_again = generate_copy_batch(
        symbol_count=8,
        memorize_length=10,
        gap=20,
        batch_size=8,
        seed=99,
        dtype=torch.float32,
    )
    copy64 = generate_copy_batch(
        symbol_count=8,
        memorize_length=10,
        gap=20,
        batch_size=8,
        seed=99,
        dtype=torch.float64,
    )
    adding32 = generate_adding_batch(
        length=20, batch_size=8, seed=99, dtype=torch.float32
    )
    adding64 = generate_adding_batch(
        length=20, batch_size=8, seed=99, dtype=torch.float64
    )

    assert torch.equal(copy32.inputs, copy32_again.inputs)
    assert torch.equal(copy32.targets, copy32_again.targets)
    assert torch.equal(copy64.inputs, copy32.inputs.to(torch.float64))
    assert torch.equal(copy64.targets, copy32.targets)
    assert torch.equal(adding64.inputs, adding32.inputs.to(torch.float64))
    assert torch.equal(adding64.targets, adding32.targets.to(torch.float64))


def test_closed_form_memoryless_baselines() -> None:
    assert copy_memoryless_loss(8) == math.log(8)
    assert adding_mean_baseline_mse() == pytest.approx(1.0 / 6.0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"symbol_count": 1}, "at least two"),
        ({"memorize_length": 0}, "positive integer"),
        ({"gap": -1}, "non-negative integer"),
        ({"batch_size": 0}, "positive integer"),
        ({"seed": -1}, "non-negative integer"),
    ],
)
def test_invalid_copy_settings_are_rejected(
    overrides: dict[str, int], message: str
) -> None:
    settings: dict[str, Any] = {
        "symbol_count": 8,
        "memorize_length": 10,
        "gap": 20,
        "batch_size": 8,
        "seed": 0,
        "dtype": torch.float32,
    }
    settings.update(overrides)
    with pytest.raises(ValueError, match=message):
        generate_copy_batch(**settings)


def test_invalid_adding_settings_and_dtypes_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least two"):
        generate_adding_batch(length=1, batch_size=8, seed=0, dtype=torch.float32)
    with pytest.raises(ValueError, match="audit dtype"):
        generate_adding_batch(length=2, batch_size=8, seed=0, dtype=torch.float16)
    with pytest.raises(ValueError, match="audit dtype"):
        generate_copy_batch(
            symbol_count=8,
            memorize_length=10,
            gap=20,
            batch_size=8,
            seed=0,
            dtype=torch.float16,
        )
