#!/usr/bin/env bash
set -euo pipefail

foundation_paths=(
  spbunn/reporting
  spbunn/utils/common.py
  tests
)

black --check "${foundation_paths[@]}"
isort --check-only "${foundation_paths[@]}"
pylint "${foundation_paths[@]}"
mypy "${foundation_paths[@]}"
