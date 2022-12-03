#!/usr/bin/env bash

set -x

CODEBASE="$(dirname "$(dirname "$0")")"
CODEBASE="$(realpath "$CODEBASE")"

SEEDS="80 444 59"

PYTHONPATH=$CODEBASE \
  python "$CODEBASE/scripts/midway_report_exps.py" --seeds ${SEEDS} \
  --experiments 5slab_sideprob 7slab_sideprob 5slab_inputdim 7slab_inputdim \
  --no-save-data
