"""Copy-task RNN and LSTM models with packed and explicit recurrences."""

from dataclasses import dataclass
from typing import Literal, cast, overload

import torch
from torch import nn
from torch.nn import functional as F

from spbunn.rnn.tasks import SUPPORTED_DTYPES

ModelKind = Literal["rnn", "lstm"]
ModelInitialization = Literal["default", "orthogonal"]


@dataclass(frozen=True)
class RNNUnroll:
    """Separate hidden states and logits from an explicit RNN recurrence."""

    states: tuple[torch.Tensor, ...]
    logits: torch.Tensor


@dataclass(frozen=True)
class LSTMUnroll:
    """Separate cell states, forget gates, and logits from an LSTM recurrence."""

    cell_states: tuple[torch.Tensor, ...]
    forget_gates: tuple[torch.Tensor, ...]
    logits: torch.Tensor


class CopyRNNModel(nn.Module):
    """A packed tanh RNN with an equivalent explicitly unrolled audit path."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.recurrent = nn.RNN(
            input_size, hidden_size, batch_first=True, nonlinearity="tanh"
        )
        self.readout = nn.Linear(hidden_size, input_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return packed readout logits for efficient optimization."""
        states, _ = self.recurrent(inputs)
        return self.readout(states)

    @property
    def weight_ih(self) -> torch.Tensor:
        """Return the input-to-hidden matrix with a static tensor type."""
        return cast(torch.Tensor, self.recurrent.weight_ih_l0)

    @property
    def weight_hh(self) -> torch.Tensor:
        """Return the hidden-to-hidden matrix with a static tensor type."""
        return cast(torch.Tensor, self.recurrent.weight_hh_l0)

    @property
    def bias_ih(self) -> torch.Tensor:
        """Return the input-side bias with a static tensor type."""
        return cast(torch.Tensor, self.recurrent.bias_ih_l0)

    @property
    def bias_hh(self) -> torch.Tensor:
        """Return the recurrent-side bias with a static tensor type."""
        return cast(torch.Tensor, self.recurrent.bias_hh_l0)

    def unroll(self, inputs: torch.Tensor, *, retain_states: bool = True) -> RNNUnroll:
        """Apply RNN parameters step by step and expose every hidden state."""
        _validate_unroll_inputs(inputs, self.recurrent.input_size)
        hidden = inputs.new_zeros(inputs.shape[0], self.recurrent.hidden_size)
        states = []
        for time_index in range(inputs.shape[1]):
            hidden = (
                F.linear(
                    inputs[:, time_index],
                    self.weight_ih,
                    self.bias_ih,
                )
                + F.linear(hidden, self.weight_hh, self.bias_hh)
            ).tanh()
            if retain_states and hidden.requires_grad:
                hidden.retain_grad()
            states.append(hidden)
        logits = self.readout(torch.stack(states, dim=1))
        return RNNUnroll(states=tuple(states), logits=logits)


class CopyLSTMModel(nn.Module):
    """A packed LSTM with explicit GetSomeMath-compatible cell equations."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.recurrent = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.readout = nn.Linear(hidden_size, input_size)
        self.zero_forget_bias()

    @torch.no_grad()
    def zero_forget_bias(self) -> None:
        """Set both PyTorch forget-bias vectors to zero."""
        hidden_size = self.recurrent.hidden_size
        self.bias_ih[hidden_size : 2 * hidden_size].zero_()
        self.bias_hh[hidden_size : 2 * hidden_size].zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return packed readout logits for efficient optimization."""
        states, _ = self.recurrent(inputs)
        return self.readout(states)

    @property
    def weight_ih(self) -> torch.Tensor:
        """Return the packed four-gate input matrix with a static tensor type."""
        return cast(torch.Tensor, self.recurrent.weight_ih_l0)

    @property
    def weight_hh(self) -> torch.Tensor:
        """Return the packed four-gate recurrent matrix with a static tensor type."""
        return cast(torch.Tensor, self.recurrent.weight_hh_l0)

    @property
    def bias_ih(self) -> torch.Tensor:
        """Return the packed input-side gate biases with a static tensor type."""
        return cast(torch.Tensor, self.recurrent.bias_ih_l0)

    @property
    def bias_hh(self) -> torch.Tensor:
        """Return the packed recurrent-side gate biases with a static tensor type."""
        return cast(torch.Tensor, self.recurrent.bias_hh_l0)

    def unroll(self, inputs: torch.Tensor, *, retain_states: bool = True) -> LSTMUnroll:
        """Apply LSTM parameters step by step and expose cells and forget gates."""
        _validate_unroll_inputs(inputs, self.recurrent.input_size)
        batch_size = inputs.shape[0]
        hidden_size = self.recurrent.hidden_size
        hidden = inputs.new_zeros(batch_size, hidden_size)
        cell = inputs.new_zeros(batch_size, hidden_size)
        cell_states = []
        forget_gates = []
        hidden_states = []
        for time_index in range(inputs.shape[1]):
            gates = F.linear(
                inputs[:, time_index],
                self.weight_ih,
                self.bias_ih,
            ) + F.linear(hidden, self.weight_hh, self.bias_hh)
            input_gate, forget_gate, candidate, output_gate = gates.chunk(4, dim=-1)
            input_gate = input_gate.sigmoid()
            forget_gate = forget_gate.sigmoid()
            candidate = candidate.tanh()
            output_gate = output_gate.sigmoid()
            cell = forget_gate * cell + input_gate * candidate
            hidden = output_gate * cell.tanh()
            if retain_states and cell.requires_grad:
                cell.retain_grad()
            cell_states.append(cell)
            forget_gates.append(forget_gate)
            hidden_states.append(hidden)
        logits = self.readout(torch.stack(hidden_states, dim=1))
        return LSTMUnroll(
            cell_states=tuple(cell_states),
            forget_gates=tuple(forget_gates),
            logits=logits,
        )


@overload
def build_copy_model(
    *,
    kind: Literal["rnn"],
    initialization: ModelInitialization,
    seed: int,
    hidden_size: int,
    input_size: int,
    dtype: torch.dtype = torch.float32,
) -> CopyRNNModel: ...


@overload
def build_copy_model(
    *,
    kind: Literal["lstm"],
    initialization: ModelInitialization,
    seed: int,
    hidden_size: int,
    input_size: int,
    dtype: torch.dtype = torch.float32,
) -> CopyLSTMModel: ...


def build_copy_model(
    *,
    kind: ModelKind,
    initialization: ModelInitialization,
    seed: int,
    hidden_size: int,
    input_size: int,
    dtype: torch.dtype = torch.float32,
) -> CopyRNNModel | CopyLSTMModel:
    """Construct an isolated, seeded copy model and then convert its dtype."""
    if kind not in {"rnn", "lstm"}:
        raise ValueError(f"Unsupported copy-model kind: {kind}")
    if initialization not in {"default", "orthogonal"}:
        raise ValueError(f"Unsupported copy-model initialization: {initialization}")
    if kind == "lstm" and initialization != "default":
        raise ValueError("The LSTM arm requires default non-forget initialization")
    _positive_integer(hidden_size, "The hidden size")
    _positive_integer(input_size, "The input size")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("The model seed must be a non-negative integer")
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError("The copy-model dtype must be torch.float32 or torch.float64")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if kind == "rnn":
            model: CopyRNNModel | CopyLSTMModel = CopyRNNModel(input_size, hidden_size)
            if initialization == "orthogonal":
                nn.init.orthogonal_(model.weight_hh)
        else:
            model = CopyLSTMModel(input_size, hidden_size)
    model.to(dtype=dtype)
    return model


def parameter_count(model: CopyRNNModel | CopyLSTMModel) -> int:
    """Return the number of trainable scalar parameters."""
    return sum(parameter.numel() for parameter in model.parameters())


def _validate_unroll_inputs(inputs: torch.Tensor, input_size: int) -> None:
    if inputs.ndim != 3:
        raise ValueError("Unroll inputs must have batch, time, and feature dimensions")
    if inputs.shape[-1] != input_size:
        raise ValueError("The unroll input feature dimension does not match the model")


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
