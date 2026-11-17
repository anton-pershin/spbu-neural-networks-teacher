#!/usr/bin/env bash
set -euo pipefail

checked_paths=(
  spbunn/optimization/landscape.py
  spbunn/optimization/mini_batch.py
  spbunn/optimization/plotting.py
  spbunn/optimization/schedulers.py
  spbunn/reporting
  spbunn/scripts/loss_landscape.py
  spbunn/scripts/schedulers.py
  spbunn/scripts/simple_loss_optimization.py
  spbunn/utils/common.py
  tests
)

black --check "${checked_paths[@]}"
isort --check-only "${checked_paths[@]}"
pylint "${checked_paths[@]}"
mypy "${checked_paths[@]}"
