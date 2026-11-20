"""Static prediction examples and inline HTML animation controls."""

import json
from html import escape

from spbunn.rnn.examples import (
    AddingPredictionExample,
    CopyPredictionExample,
    PredictionExamples,
)


def build_copy_prediction_example(example: CopyPredictionExample) -> str:
    """Build copy input/target/output tables and its animation container."""
    frames = _copy_frames(example)
    reproduction_start = len(example.inputs) - len(example.targets)
    input_text = ", ".join(
        _copy_token(token, example.symbol_count) for token in example.inputs
    )
    target_text = ", ".join(
        _copy_token(token, example.symbol_count) for token in example.targets
    )
    output_text = ", ".join(
        _copy_token(token, example.symbol_count)
        for token in example.predictions[reproduction_start:]
    )
    return _example_section(
        task="copy",
        heading="Copy task: input, target, and RNN output example",
        description=(
            "The input contains ordinary symbol IDs, a delimiter, and blanks. "
            "Loss applies only at the reproduction positions. The displayed "
            "RNN output is the argmax class at those positions from the "
            f"untrained default model with seed {example.model_seed}."
        ),
        input_text=input_text,
        target_text=target_text,
        output_text=output_text,
        frames=frames,
    )


def build_adding_prediction_example(example: AddingPredictionExample) -> str:
    """Build adding input/target/output tables and its animation container."""
    frames = _adding_frames(example)
    input_text = ", ".join(
        f"({value:.3f}, {marker})"
        for value, marker in zip(example.values, example.markers, strict=True)
    )
    return _example_section(
        task="adding",
        heading="Adding task: input, target, and RNN output example",
        description=(
            "Each input pair below is (random value, marker). Exactly two "
            "markers are one. The target is the sum of their values. The "
            "displayed RNN output is the final scalar from the untrained "
            f"Xavier-tanh model with seed {example.model_seed}."
        ),
        input_text=input_text,
        target_text=f"{example.target:.6f}",
        output_text=f"{example.predictions[-1]:.6f}",
        frames=frames,
    )


def build_animation_script(examples: PredictionExamples, *, interval_ms: int) -> str:
    """Return inline CSS and JavaScript for play/pause/previous/next controls."""
    payload = {
        "copy": {
            "frames": _copy_frames(examples.copy),
            "interval": interval_ms,
        },
        "adding": {
            "frames": _adding_frames(examples.adding),
            "interval": interval_ms,
        },
    }
    return f"""
<style>
.rnn-animation {{ margin: 1rem 0 2rem; }}
.animation-controls {{ display: flex; gap: 0.5rem; flex-wrap: wrap; }}
.animation-controls button {{ padding: 0.4rem 0.8rem; }}
.animation-status {{ font-weight: 600; }}
.rnn-animation table {{ margin-top: 0.6rem; }}
.rnn-animation td, .rnn-animation th {{ padding: 0.35rem 0.6rem; }}
</style>
<script>
(() => {{
  const configs = {_safe_json(payload)};
  const states = {{}};
  const byId = (id) => document.getElementById(id);

  function render(name) {{
    const state = states[name];
    const config = configs[name];
    const frame = config.frames[state.index];
    byId(`${{name}}-step`).textContent = String(state.index + 1);
    byId(`${{name}}-input`).textContent = frame.input;
    byId(`${{name}}-prediction`).textContent = frame.prediction;
    byId(`${{name}}-target`).textContent = frame.target;
    byId(`${{name}}-status`).textContent =
      `Step ${{state.index + 1}} of ${{config.frames.length}}`;
    byId(`${{name}}-play`).disabled = state.timer !== null;
    byId(`${{name}}-pause`).disabled = state.timer === null;
    byId(`${{name}}-prev`).disabled = state.index === 0;
    byId(`${{name}}-next`).disabled = state.index === config.frames.length - 1;
  }}

  function pause(name) {{
    const state = states[name];
    if (state.timer !== null) {{
      clearInterval(state.timer);
      state.timer = null;
    }}
    render(name);
  }}

  function play(name) {{
    const state = states[name];
    const config = configs[name];
    if (state.timer !== null) return;
    if (state.index === config.frames.length - 1) state.index = 0;
    state.timer = setInterval(() => {{
      if (state.index === config.frames.length - 1) {{
        pause(name);
      }} else {{
        state.index += 1;
        render(name);
      }}
    }}, config.interval);
    render(name);
  }}

  function step(name, delta) {{
    const state = states[name];
    pause(name);
    state.index = Math.max(
      0, Math.min(configs[name].frames.length - 1, state.index + delta)
    );
    render(name);
  }}

  Object.keys(configs).forEach((name) => {{
    states[name] = {{ index: 0, timer: null }};
    byId(`${{name}}-play`).addEventListener("click", () => play(name));
    byId(`${{name}}-pause`).addEventListener("click", () => pause(name));
    byId(`${{name}}-prev`).addEventListener("click", () => step(name, -1));
    byId(`${{name}}-next`).addEventListener("click", () => step(name, 1));
    render(name);
  }});
}})();
</script>"""


def _copy_frames(example: CopyPredictionExample) -> list[dict[str, str]]:
    target_start = len(example.inputs) - len(example.targets)
    frames = []
    for index, (token, prediction) in enumerate(
        zip(example.inputs, example.predictions, strict=True)
    ):
        target = "—"
        if index >= target_start:
            target = _copy_token(
                example.targets[index - target_start], example.symbol_count
            )
        frames.append(
            {
                "input": _copy_token(token, example.symbol_count),
                "prediction": _copy_token(prediction, example.symbol_count),
                "target": target,
            }
        )
    return frames


def _adding_frames(example: AddingPredictionExample) -> list[dict[str, str]]:
    final_index = len(example.values) - 1
    return [
        {
            "input": f"value={value:.6f}, marker={marker}",
            "prediction": f"{prediction:.6f}",
            "target": f"{example.target:.6f}" if index == final_index else "—",
        }
        for index, (value, marker, prediction) in enumerate(
            zip(
                example.values,
                example.markers,
                example.predictions,
                strict=True,
            )
        )
    ]


def _example_section(
    *,
    task: str,
    heading: str,
    description: str,
    input_text: str,
    target_text: str,
    output_text: str,
    frames: list[dict[str, str]],
) -> str:
    first = frames[0]
    return f"""
<section class="prediction-example">
<h3>{escape(heading)}</h3>
<p>{escape(description)}</p>
<table><tbody>
<tr><th>Input sequence</th><td><code>{escape(input_text)}</code></td></tr>
<tr><th>Target output</th><td><code>{escape(target_text)}</code></td></tr>
<tr><th>Untrained RNN output</th><td><code>{escape(output_text)}</code></td></tr>
</tbody></table>
<p>Use the controls to inspect every time step. Predictions can be poor because
this model is untrained; the animation shows behavior, not task quality.</p>
<div class="rnn-animation" id="{task}-animation">
<div class="animation-controls">
<button type="button" id="{task}-play">Play</button>
<button type="button" id="{task}-pause" disabled>Pause</button>
<button type="button" id="{task}-prev" disabled>Previous</button>
<button type="button" id="{task}-next">Next</button>
</div>
<p class="animation-status" id="{task}-status"
 aria-live="polite">Step 1 of {len(frames)}</p>
<table><thead><tr><th>Step</th><th>Input</th><th>RNN prediction</th><th>Target</th></tr></thead>
<tbody><tr>
<td id="{task}-step">1</td>
<td id="{task}-input">{escape(first["input"])}</td>
<td id="{task}-prediction">{escape(first["prediction"])}</td>
<td id="{task}-target">{escape(first["target"])}</td>
</tr></tbody></table>
</div>
</section>"""


def _copy_token(token: int, symbol_count: int) -> str:
    if token < symbol_count:
        return str(token)
    if token == symbol_count:
        return "<blank>"
    return "<delimiter>"


def _safe_json(payload: object) -> str:
    return (
        json.dumps(payload, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
