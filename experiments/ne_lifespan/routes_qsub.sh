#!/usr/bin/env bash
#$ -N routes
#$ -cwd
# $JOB_ID/$TASK_ID are SGE pseudo-variables and DO interpolate here; ordinary
# environment variables do NOT -- an earlier version used $PHASE and produced files
# literally named 'routes$PHASE.1.out'. $JOB_ID keeps each submission's logs separate,
# so phase 2 cannot overwrite phase 1's. (oscillation_qsub.sh has the same latent bug.)
#$ -o logs/routes.$JOB_ID.$TASK_ID.out
#$ -e logs/routes.$JOB_ID.$TASK_ID.err
#$ -l h_vmem=16G
#$ -V
#$ -t 1-3
# SGE array for the three-route experiment: does ecology reach life history through the
# drift barrier (Ne), through mutational supply (N*u), or through extrinsic mortality?
# See routes_configs.py for the argument; HANDOFF.md for the calibration behind the grid.
#
# gen100 is the HEAD/LOGIN node -- submit from it, never run anything on it.
#
#   PHASE 1  lattice burn-in at the MOST MIXED setting, one job per seed.
#   PHASE 2  9 arms x n_seeds, each resuming its OWN COPY of the seed's ancestor with
#            one parameter overridden.
#
#   cd ~/aegis
#   CONFIG_DIR=/wins/vlzno/projects/aegis_routes
#   python experiments/ne_lifespan/routes_configs.py --outdir $CONFIG_DIR
#   mkdir -p logs
#   CONFIG_DIR=$CONFIG_DIR PHASE=1 qsub -t 1-3 experiments/ne_lifespan/routes_qsub.sh
#   # WAIT, then CHECK -- do not skip:
#   python runs/check_equilibration.py $CONFIG_DIR/burn_s1
#   CONFIG_DIR=$CONFIG_DIR TOTAL=300000 PHASE=2 qsub -t 1-27 experiments/ne_lifespan/routes_qsub.sh
#
# PHASE must be EXPORTED at submit time (as above): both phases share task numbers and
# #$ -V is what carries PHASE into the log filename.
#
# WALL CLOCK. Measured in the calibration: ~20-60 ms/step at K=3000 with lattice on (the
# spatial model does per-individual placement in Python, so it is slower than well-mixed).
# 100k burn-in ~ 1-2 h; 200k released ~ 2-4 h. CHECKPOINT_RATE=10000 means a killed job
# resumes -- just resubmit the same array.
#
# EACH ARM NEEDS ITS OWN BYTE COPY of the ancestor. Resume opens the output CSVs in APPEND
# mode (recordingmanager.init_for_resume), so a hard-linked copy would share an inode with
# the ancestor and all 9 arms would append into one file at once. cp -r, never cp -al.
# This is the same trap documented in oscillation_qsub.sh.
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

# VERIFIED 2026-08-15 on gen100: this env exists, holds bin/aegis, and its aegis_sim is an
# EDITABLE install pointing at ~/aegis -- so a git checkout there updates the engine, no
# pip install needed. `conda activate aegis` before qsub anyway so #$ -V carries it too.
AEGIS_ENV="${AEGIS_ENV:-/home/lakatos/dvalenza/.conda/envs/aegis}"
export PATH="${AEGIS_ENV}/bin:${PATH}"

if ! command -v aegis >/dev/null 2>&1; then
    echo "ERROR: 'aegis' not on PATH. AEGIS_ENV=${AEGIS_ENV}"
    exit 1
fi

TASK_ID="${SGE_TASK_ID:-1}"
PHASE="${PHASE:-1}"
CONFIG_DIR="${CONFIG_DIR:-/wins/vlzno/projects/aegis_routes}"
# TOTAL steps, not an increment: `--extend N` sets STEPS_PER_SIMULATION = N outright
# (aegis_sim.init_resume), and it REJECTS N <= the checkpoint step. So this must be
# burn_steps + fwd_steps. routes_configs.py prints the right number; do not pass the
# forward count alone or every arm is silently truncated.
TOTAL="${TOTAL:-300000}"
BURN_TOTAL="${BURN_TOTAL:-}"   # phase 1 only: extend a finished burn-in to this many TOTAL steps

# Arm name -> the single --override that defines it. MUST match ARMS in routes_configs.py.
# Order fixes the phase-2 task mapping, so do not reorder without re-reading the -t range.
# C_starv turned out NOT to test extrinsic mortality: ARCH sets STARVATION_PENALTY=0.0,
# and the multiplier is (1-penalty)**steps = 1.0 always, so removing the birth cap left
# nothing regulating the population at all -- it grew to 9885 (3.3x K) with no penalty.
# It is retained as a genuine high-N arm (an extra route-1 point, +2.12 in e0), and
# C_starv_pen is the CORRECTED route-3 arm with the penalty actually switched on.
ARM_NAMES=(ctrl A_ld0200 A_ld0050 A_ld0010 A_ld0000 B_mu05 B_mu20 B_mu40 C_starv C_starv_pen)
ARM_OVERRIDE=(
    ""
    "MIGRATION_LONG_RATE=0.02"
    "MIGRATION_LONG_RATE=0.005"
    "MIGRATION_LONG_RATE=0.001"
    "MIGRATION_LONG_RATE=0.0"
    "G_muta_initpheno=8.5e-05"
    "G_muta_initpheno=0.00034"
    "G_muta_initpheno=0.00068"
    "REPRODUCTION_REGULATION=false"
    "REPRODUCTION_REGULATION=false --override STARVATION_PENALTY=0.1"
)

echo "host: $(hostname) | phase: ${PHASE} | task: ${TASK_ID} | started: $(date)"

if [ "${PHASE}" = "1" ]; then
    # STRICT glob: match burn_s<N>.yml ONLY. Phase 2 writes an arm config beside every
    # run (burn_s1_ctrl.yml, burn_s1_A_ld0000.yml, ...), so a bare burn_s*.yml glob
    # silently starts resolving ARM configs as ancestors once any arm has run -- which
    # is exactly what killed killifish tasks 10-27 on 2026-08-16.
    CONFIG=$(ls "${CONFIG_DIR}"/burn_s*.yml 2>/dev/null | grep -E "/burn_s[0-9]+\.yml$" | sed -n "${TASK_ID}p")
    if [ -z "${CONFIG}" ]; then
        echo "no burn-in config for task ${TASK_ID} in ${CONFIG_DIR}"
        echo "  generate: python experiments/ne_lifespan/routes_configs.py --outdir ${CONFIG_DIR}"
        exit 1
    fi
    NAME=$(basename "${CONFIG}" .yml)
    OUTDIR="${CONFIG_DIR}/${NAME}"
    echo "run: ${NAME} (lattice burn-in, most-mixed regime)"

    # BURN_TOTAL lets a FINISHED burn-in be pushed further without losing it, e.g.
    #   BURN_TOTAL=400000 PHASE=1 qsub -t 1-3 ...
    # Needed because the neutral-locus relaxation time scales as -ln(tol)/mu generations,
    # and at G_muta_initpheno=1.7e-4 that could be ~27k generations (~400-550k steps) --
    # far more than the 100k default. Do not guess: run check_equilibration.py, read the
    # "gap left" column, and extend only if it says so. --extend is a TOTAL.
    if [ -n "${BURN_TOTAL}" ] && [ -d "${OUTDIR}" ] && [ -e "${OUTDIR}/checkpoint" ]; then
        echo "extending burn-in to ${BURN_TOTAL} TOTAL steps"
        rm -f "${OUTDIR}/.phase1_done"
        aegis sim -c "${CONFIG}" -r --extend "${BURN_TOTAL}"
        STATUS=$?
        [ ${STATUS} -eq 0 ] && [ -f "${OUTDIR}/output_summary.json" ] && touch "${OUTDIR}/.phase1_done"
        echo "finished: $(date) | exit: ${STATUS}"
        exit ${STATUS}
    fi
    if [ -f "${OUTDIR}/.phase1_done" ]; then
        echo "already complete -- skipping"; exit 0
    fi
    if [ -d "${OUTDIR}" ] && [ ! -e "${OUTDIR}/checkpoint" ]; then
        echo "output dir with no checkpoint (killed early) -> discarding and restarting"
        rm -rf "${OUTDIR}"
    fi
    if [ -d "${OUTDIR}" ]; then
        echo "resuming from latest checkpoint"; aegis sim -c "${CONFIG}" -r
    else
        echo "fresh start"; aegis sim -c "${CONFIG}"
    fi
    STATUS=$?

    # The lattice must actually have engaged, or every downstream arm is a well-mixed
    # impostor. Catch it here rather than after 27 phase-2 jobs.
    if [ ${STATUS} -eq 0 ] && { [ ! -d "${OUTDIR}/lattice" ] || \
         [ -z "$(ls -A "${OUTDIR}/lattice" 2>/dev/null)" ]; }; then
        echo "ERROR: ${NAME} wrote no lattice snapshots -- the spatial model did not engage."
        STATUS=1
    fi
    [ ${STATUS} -eq 0 ] && [ -f "${OUTDIR}/output_summary.json" ] && touch "${OUTDIR}/.phase1_done"

else
    N_ARMS=${#ARM_NAMES[@]}
    SEED_IDX=$(( (TASK_ID - 1) / N_ARMS ))
    ARM=$(( (TASK_ID - 1) % N_ARMS ))
    ARM_NAME=${ARM_NAMES[$ARM]}
    OVERRIDE=${ARM_OVERRIDE[$ARM]}

    CONFIG=$(ls "${CONFIG_DIR}"/burn_s*.yml 2>/dev/null | grep -E "/burn_s[0-9]+\.yml$" | sed -n "$(( SEED_IDX + 1 ))p")
    if [ -z "${CONFIG}" ]; then
        echo "no burn-in config for seed index ${SEED_IDX} -- is -t larger than n_arms*n_seeds?"
        exit 1
    fi
    NAME=$(basename "${CONFIG}" .yml)
    OUTDIR="${CONFIG_DIR}/${NAME}"
    if [ ! -f "${OUTDIR}/.phase1_done" ]; then
        echo "ERROR: ${NAME} has not finished phase 1. Run PHASE=1 first, and CHECK"
        echo "       equilibration: python runs/check_equilibration.py ${OUTDIR}"
        exit 1
    fi

    ARMDIR="${CONFIG_DIR}/${NAME}_${ARM_NAME}"
    if [ -f "${ARMDIR}/output_summary.json" ] && [ -f "${ARMDIR}/.arm_done" ]; then
        echo "already complete -- skipping"; exit 0
    fi
    # Real byte copy per arm; NEVER cp -al (see header). Stagger so 27 arms do not hit
    # the network mount at once.
    if [ ! -d "${ARMDIR}" ]; then
        sleep $(( (TASK_ID % N_ARMS) * 3 ))
        cp -r "${OUTDIR}" "${ARMDIR}" || exit 1
        rm -f "${ARMDIR}/.phase1_done"
    fi
    cp "${CONFIG}" "${ARMDIR}.yml"

    echo "run: $(basename "${ARMDIR}")  arm=${ARM_NAME}  override='${OVERRIDE:-none}'  extend-to=${TOTAL} (TOTAL steps, incl. burn-in)"
    if [ -n "${OVERRIDE}" ]; then
        # ${OVERRIDE} is deliberately UNQUOTED: an arm may carry several settings as
        # "A=1 --override B=2", and word-splitting is what turns that into separate
        # arguments. Quoting it passes the whole string as one --override value, which
        # fails as "expects a boolean, got 'false --override STARVATION_PENALTY=0.1'".
        # Harmless for single-setting arms, which is why it survived until arm C_starv_pen.
        # shellcheck disable=SC2086
        aegis sim -c "${ARMDIR}.yml" -r --extend "${TOTAL}" --override ${OVERRIDE}
    else
        aegis sim -c "${ARMDIR}.yml" -r --extend "${TOTAL}"
    fi
    STATUS=$?
    [ ${STATUS} -eq 0 ] && [ -f "${ARMDIR}/output_summary.json" ] && touch "${ARMDIR}/.arm_done"
fi

echo "finished: $(date) | exit: ${STATUS}"
exit ${STATUS}
