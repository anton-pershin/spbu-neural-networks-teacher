#!/usr/bin/env bash
set -euo pipefail

checked_paths=(
  spbunn/convolution.py
  spbunn/data/__init__.py
  spbunn/mlp/architecture.py
  spbunn/mlp/propagation.py
  spbunn/ngram
  spbunn/optimization/landscape.py
  spbunn/optimization/mini_batch.py
  spbunn/optimization/plotting.py
  spbunn/optimization/schedulers.py
  spbunn/reporting
  spbunn/rnn
  spbunn/scripts/balduzzi_analysis.py
  spbunn/scripts/bptt_instability.py
  spbunn/scripts/gradient_flow_audit.py
  spbunn/scripts/bpe.py
  spbunn/scripts/hidden_norms_propagation.py
  spbunn/scripts/loss_landscape.py
  spbunn/scripts/mlp_depth_experiment.py
  spbunn/scripts/mlp_width_experiment.py
  spbunn/scripts/ngram.py
  spbunn/scripts/schedulers.py
  spbunn/scripts/simple_convolution.py
  spbunn/scripts/simple_loss_optimization.py
  spbunn/sequence
  spbunn/utils/common.py
  tests
)

black --check "${checked_paths[@]}"
isort --check-only "${checked_paths[@]}"
pylint "${checked_paths[@]}"
mypy "${checked_paths[@]}"
