#!/usr/bin/env bash
# Capacity probe: static-routing sweep (sizing for M4). PRECISIONS="bf16 fp8" to compare.
set -euo pipefail
for p in ${PRECISIONS:-bf16}; do
  PHASE=probe PRECISION=$p SWEEP_S=${SWEEP_S:-90} bash scripts/m4_load.sh
done
