#!/usr/bin/env bash
# grow_horizon.sh <EXPT> — fit the FULL record via a warm-started curriculum.
#
# A from-scratch MAP fit of a long, many-cycle record collapses to the flat mean
# (the optimiser never finds the cycle basin). This marches there in stages: fit a
# short window (cycle found), then warm-start each longer window from the previous
# stage's weights, up to the full record. Days-per-ODE-time-unit is held constant
# (DPU) so the learned vector field f_θ transfers across stages. Best-val
# checkpointing (BEST_VAL=1) keeps each stage's best-forecast weights.
#
# Run from paper-experiments/AISTATS/algae-chemostat/ :
#   bash grow_horizon.sh 6
# (no `set -e`: one stage hiccuping should not abort the whole ladder)

EXPT=${1:-6}
DPU=${DPU:-7.5}                 # days per ODE-time unit (constant across stages)
A=${MAP_PHASEA:-4000}; B=${MAP_PHASEB:-200}
PARAMS=outputs/algae_chemostat/map_params_C${EXPT}.csv
DATA=data/blasius_rotifer_algae.csv

MAXDAY=$(python3 -c "import csv;ds=[float(r['day']) for r in csv.DictReader(open('$DATA')) if int(r['experiment'])==$EXPT];print(int(max(ds)))")
echo "C$EXPT spans 0–$MAXDAY days. Curriculum (DPU=$DPU days/unit):"

# Ladder of window lengths (days); final rung = full record. Drop rungs >= MAXDAY.
STAGES=()
for d in 60 120; do [ "$d" -lt "$MAXDAY" ] && STAGES+=("$d"); done
STAGES+=("$MAXDAY")
echo "  stages: ${STAGES[*]}"

i=0
for DM in "${STAGES[@]}"; do
  i=$((i+1))
  TMAX=$(python3 -c "print(round($DM/$DPU,2))")
  # Warm-start every stage after the first (env passes an *expanded* assignment,
  # unlike a bare `$INIT cmd` prefix which bash won't re-parse as an assignment).
  if [ "$i" -eq 1 ]; then INIT=""; MODE="random-init"; else INIT="INIT_FROM=$PARAMS"; MODE="warm-start"; fi
  echo
  echo "===== stage $i/${#STAGES[@]}: C$EXPT  DAY_MAX=$DM  TMAX=$TMAX  $MODE ====="
  rm -f outputs/algae_chemostat/map_faceted.png      # so a failed stage can't leave a stale plot
  env MAP_ONLY=1 EXPT=$EXPT DAY_MAX=$DM TMAX=$TMAX MAP_PHASEA=$A MAP_PHASEB=$B \
      BEST_VAL=1 NWIN=7 SAVE_PARAMS=$PARAMS $INIT \
      caffeinate -i julia --project=../../.. algae_chemostat.jl
  if [ -f outputs/algae_chemostat/map_faceted.png ]; then
    cp outputs/algae_chemostat/map_faceted.png outputs/algae_chemostat/map_faceted_C${EXPT}_dm${DM}.png
    echo "  → saved fit plot: outputs/algae_chemostat/map_faceted_C${EXPT}_dm${DM}.png"
  else
    echo "  !! stage $i produced no fit plot — Julia likely errored; stopping ladder." ; break
  fi
done

echo
echo "DONE. Final full-record fit weights: $PARAMS"
echo "Inspect the ladder: outputs/algae_chemostat/map_faceted_C${EXPT}_dm*.png"
