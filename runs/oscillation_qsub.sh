#!/usr/bin/env bash
#$ -N oscph
#$ -cwd
#$ -o logs/oscph$PHASE.$TASK_ID.out
#$ -e logs/oscph$PHASE.$TASK_ID.err
#$ -l h_vmem=8G
#$ -V
#$ -t 1-3
# SGE array wrapper for the two-phase resource-oscillation experiment.
#
# gen100 is the HEAD/LOGIN node -- submit from it, never run anything on it.
#
# WHY TWO PHASES. Anything measured from initialization is contaminated by the burn-in
# transient, so the experiment must start from an equilibrated population. The neutral
# locus gives an objective stopping criterion: carrying no phenotypic effect, it relaxes
# under mutation and drift alone toward p* = MUTATION_RATIO/(1+MUTATION_RATIO) = 0.0909,
# geometrically per GENERATION, and p* does not depend on the mutation rate. Once the
# neutral load sits at p*, initial conditions are forgotten.
#
# PHASE 1 -- burn in under constant, regulated resources (N pinned, no starvation):
#   python runs/resource_oscillation_configs.py --outdir $CONFIG_DIR --burnin \
#          --seeds 1 2 3 --steps 200000
#   mkdir -p logs
#   PHASE=1 qsub -t 1-3 runs/oscillation_qsub.sh
#   # then CHECK equilibration before going on -- do not skip this:
#   python runs/check_equilibration.py $CONFIG_DIR/burnin_R2000_s1
#
# PHASE 2 -- branch the scan off each equilibrated checkpoint. Every arm resumes the
# SAME ancestor, so between-arm differences cannot come from burn-in history:
#   PHASE=2 qsub -t 1-36 runs/oscillation_qsub.sh
#
# Set -t to the number of tasks: phase 1 = n_seeds, phase 2 = n_seeds * 12.
#
# Logs carry the phase in their name (oscph1.N.out / oscph2.N.out): both phases use
# the same task numbers, so a shared name lets phase 2 overwrite phase 1's record.
# PHASE must therefore be exported at submit time, not just set in the environment:
#   PHASE=2 CONFIG_DIR=... qsub -t 1-36 runs/oscillation_qsub.sh   (with #$ -V)
#
# Measured locally: ~200k burn-in steps at ~23 ms/step ~= 80 min, 26.6 steps/generation
# under regulation (17.8 without -- the cap removes starvation mortality, so individuals
# live longer). 99% neutral relaxation needs ~4,605 generations ~= 122k steps; 200k
# leaves ~5e-4 of the initial gap. Compute nodes are slower per core than the laptop --
# size wall clock generously and rely on CHECKPOINT_RATE.
#
# Single-threaded: aegis is numpy-vectorised but not parallel. CRUCIAL on the FLI nodes
# (49-112 cores): OpenBLAS/OMP/numba otherwise spawn one thread per core and pre-reserve
# per-thread buffers sized to the core count, exhausting the process before the sim's own
# arrays allocate ("OpenBLAS: Memory allocation still failed").
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

set -uo pipefail

AEGIS_ENV="${AEGIS_ENV:-/home/lakatos/dvalenza/.conda/envs/aegis}"
export PATH="${AEGIS_ENV}/bin:${PATH}"

if ! command -v aegis >/dev/null 2>&1; then
    echo "ERROR: 'aegis' not on PATH. AEGIS_ENV=${AEGIS_ENV}"
    exit 1
fi

# This experiment needs engine features added 2026-07-20. An install predating them
# runs the OLD resource rule and silently ignores --override, producing plausible but
# wrong results -- so fail loudly instead.
python - <<'PYCHECK' || exit 1
import sys
from aegis_sim.parameterization.default_parameters import DEFAULT_PARAMETERS as D
from aegis_sim.parameterization.parametermanager import ParameterManager
missing = []
if "RESOURCE_DEFICIT_CARRYOVER" not in D:
    missing.append("RESOURCE_DEFICIT_CARRYOVER parameter")
if not hasattr(ParameterManager, "STRUCTURAL_PARAMETERS"):
    missing.append("resume --override support")
if missing:
    print("ERROR: this aegis install is out of date, missing: " + ", ".join(missing))
    print("       git checkout exp-oscillation-burnin && pip install -e .")
    sys.exit(1)
print("engine check OK: carryover + resume overrides present")
PYCHECK

TASK_ID="${SGE_TASK_ID:-1}"
PHASE="${PHASE:-1}"
CONFIG_DIR="${CONFIG_DIR:-/wins/vlzno/projects/aegis_oscillation}"

# Phase-2 scan grid. k = 1 + mult; k > 1 makes resources self-reproducing (logistic when
# capped), which is the Rosenzweig-MacArthur regime -- it predicts that RAISING the cap
# destabilizes (paradox of enrichment), the opposite of the paper's claim that large Rbar
# stabilizes. k = 0 is donor-controlled: a constant inflow, no prey reproduction.
# Caps are multiples of the realized equilibrium N (~1930 for Rbar 2000).
MULTS=(0 0.05 0.2 0.5)
CAPS=(4000 10000 40000)
EXTEND="${EXTEND:-260000}"          # burn-in 200k + 60k of released dynamics

echo "host: $(hostname) | phase: ${PHASE} | task: ${TASK_ID} | started: $(date)"

if [ "${PHASE}" = "1" ]; then
    CONFIG=$(ls "${CONFIG_DIR}"/burnin_*.yml 2>/dev/null | sed -n "${TASK_ID}p")
    if [ -z "${CONFIG}" ]; then
        echo "no burn-in config for task ${TASK_ID} in ${CONFIG_DIR}"
        echo "  generate them: python runs/resource_oscillation_configs.py \\"
        echo "      --outdir ${CONFIG_DIR} --burnin --seeds 1 2 3 --steps 200000"
        exit 1
    fi
    NAME=$(basename "${CONFIG}" .yml)
    OUTDIR="${CONFIG_DIR}/${NAME}"
    echo "run: ${NAME} (burn-in, constant + regulated)"

    if [ -f "${OUTDIR}/output_summary.json" ] && [ ! -f "${OUTDIR}/.phase1_done" ]; then
        # output_summary.json is written at the END of a run, but a phase-2 resume
        # inherits it, so it alone cannot mark phase 1 complete. Use an explicit marker.
        touch "${OUTDIR}/.phase1_done"
    fi
    if [ -f "${OUTDIR}/.phase1_done" ]; then
        echo "already complete -- skipping"
        exit 0
    fi
    if [ -d "${OUTDIR}" ]; then
        echo "output exists -> resuming from latest checkpoint"
        aegis sim -c "${CONFIG}" -r
    else
        aegis sim -c "${CONFIG}"
    fi
    STATUS=$?
    [ ${STATUS} -eq 0 ] && touch "${OUTDIR}/.phase1_done"

else
    # Phase 2: task -> (seed, mult, cap). 12 arms per seed.
    N_ARMS=$(( ${#MULTS[@]} * ${#CAPS[@]} ))
    SEED_IDX=$(( (TASK_ID - 1) / N_ARMS ))
    ARM=$(( (TASK_ID - 1) % N_ARMS ))
    MULT=${MULTS[$(( ARM / ${#CAPS[@]} ))]}
    CAP=${CAPS[$(( ARM % ${#CAPS[@]} ))]}

    CONFIG=$(ls "${CONFIG_DIR}"/burnin_*.yml 2>/dev/null | sed -n "$(( SEED_IDX + 1 ))p")
    if [ -z "${CONFIG}" ]; then
        echo "no burn-in config for seed index ${SEED_IDX}"; exit 1
    fi
    NAME=$(basename "${CONFIG}" .yml)
    OUTDIR="${CONFIG_DIR}/${NAME}"

    if [ ! -f "${OUTDIR}/.phase1_done" ]; then
        echo "ERROR: ${NAME} has not finished phase 1 -- run PHASE=1 first, and CHECK"
        echo "       equilibration: python runs/check_equilibration.py ${OUTDIR}"
        exit 1
    fi

    # Each arm needs its own copy of the burnt-in state: resume writes in place, so
    # twelve arms sharing one directory would overwrite each other.
    #
    # MUST be a real byte copy, never a hard-link copy. Resume opens the output CSVs in
    # APPEND mode
    # (recordingmanager.init_for_resume), so a hard-linked CSV is the SAME inode as the
    # ancestor's: every arm would append straight into the ancestor's file, and all 12
    # arms of a seed would append into one shared file at once -- exactly the corruption
    # hard-linking was meant to avoid. The ~170 MB x 12 duplication is the price of it.
    # Stagger the copies so 36 arms do not hit the network mount simultaneously.
    ARMDIR="${CONFIG_DIR}/${NAME}_k$(echo "1 + ${MULT}" | bc)_cap${CAP}"
    if [ ! -d "${ARMDIR}" ]; then
        sleep $(( (TASK_ID % 12) * 3 ))
        cp -r "${OUTDIR}" "${ARMDIR}" || exit 1
        rm -f "${ARMDIR}/.phase1_done"
    fi
    cp "${CONFIG}" "${ARMDIR}.yml"

    echo "run: $(basename "${ARMDIR}")  mult=${MULT} (k=$(echo "1 + ${MULT}" | bc)) cap=${CAP}"
    aegis sim -c "${ARMDIR}.yml" -r --extend "${EXTEND}" \
        --override REPRODUCTION_REGULATION=false \
        --override RESOURCE_DEFICIT_CARRYOVER=true \
        --override RESOURCE_MULTIPLICATIVE_GROWTH="${MULT}" \
        --override RESOURCE_MAXIMUM_AMOUNT="${CAP}"
    STATUS=$?
fi

echo "finished: $(date) | exit: ${STATUS}"
exit ${STATUS}
