#!/usr/bin/env bash
# Ne sweep: vary carrying capacity K over ~2 orders of magnitude, birth-regulated
# (no starvation death) so N~=K and the ONLY thing changing is Ne. MA arm.
# Isolates the Ne -> mutation-load -> lifespan chain from the extrinsic-mortality channel.
set -e
cd "$(dirname "$0")"
VENV=./aegis_venv/bin/python
STEPS=${STEPS:-40000}
COMMON="AGE_LIMIT=30 MATURATION_AGE=6 STEPS_PER_SIMULATION=$STEPS MAX_OFFSPRING_NUMBER=3 \
  STARVATION_PENALTY=0.0 REPRODUCTION_REGULATION=True \
  POPGENSTATS_RATE=2000 POPGENSTATS_SAMPLE_SIZE=100 SNAPSHOT_RATE=$STEPS"

for K in "$@"; do
  name="K${K}"
  echo "===== $name (steps=$STEPS) ====="
  rm -rf "$name" "$name.yml"
  $VENV run_sim.py "$name.yml" $COMMON \
     INITIAL_POPULATION_SIZE=$K RESOURCE_MAXIMUM_AMOUNT=$K RESOURCE_ADDITIVE_GROWTH=$K \
     2>&1 | tail -1
done
echo "ALL DONE"
