#!/usr/bin/env bash
# Capacity probe: static-routing sweeps with a BF16 and an FP8 primary (sizing for M4).
set -euo pipefail
PHASE=probe PRECISION=bf16 SWEEP_S=90 bash scripts/m4_load.sh
PHASE=probe PRECISION=fp8 SWEEP_S=90 bash scripts/m4_load.sh
