#!/usr/bin/env bash
# Multi-seed sweep for BNODE-vs-NODE forecasting comparison.
# Runs forecast_compare.jl at 5 (data_seed, init_seed) pairs and accumulates rows
# in forecasting_results.csv so you get mean ± std on val_rel_err.
#
# Usage:
#   ./sweep.sh                # 5 seeds, ~1.5 h total (≈17 min per seed)
#   FRESH=1 ./sweep.sh        # wipe the CSV first
#
# Each row in the resulting CSV records its data_seed and init_seed, so after the
# sweep you can group by `method` and aggregate across seeds with any tool.

set -euo pipefail
cd "$(dirname "$0")"

# Run from the repo root so --project=. resolves correctly.
REPO_ROOT="$(cd ../../.. && pwd)"

SEEDS=(42 7 13 99 314)

# Wipe once if FRESH=1, then NEVER again — subsequent runs append.
if [[ "${FRESH:-0}" == "1" ]]; then
  rm -f forecasting_results.csv
fi

for i in "${!SEEDS[@]}"; do
  s="${SEEDS[$i]}"
  echo ""
  echo "================================================================"
  echo " seed $((i+1))/${#SEEDS[@]}: DATA_SEED=$s INIT_SEED=$s"
  echo "================================================================"
  DATA_SEED="$s" INIT_SEED="$s" FRESH=0 \
    julia --project="$REPO_ROOT" forecast_compare.jl
done

echo ""
echo "Sweep complete. $(grep -c '^NODE\|^"BNODE' forecasting_results.csv || true) rows in:"
echo "  $(pwd)/forecasting_results.csv"
