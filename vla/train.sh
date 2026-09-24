#!/usr/bin/env bash
# Fine-tune a policy with openpi, using the configs and filters in this directory.
#
#   vla/train.sh <config> --exp-name=<name> [--fsdp-devices=<n>] [--overwrite | --resume] [...]
#
# <config> is a file stem under configs/ (place_toys_on_plate, sort_fruits_and_toys, pack_toys). All
# arguments are passed through to openpi's scripts/train.py. The script runs from vla/, so checkpoints
# land in vla/checkpoints/ (override with --checkpoint-base-dir) and relative paths in arguments
# resolve against vla/.
set -euo pipefail

# The configs' filter paths are relative to the working directory.
cd "$(dirname "${BASH_SOURCE[0]}")"

export OPENPI_CONFIG_DIR="$PWD/configs"
export GIT_LFS_SKIP_SMUDGE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

exec uv run --project ../submodules/openpi python ../submodules/openpi/scripts/train.py "$@"
