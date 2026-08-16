#!/usr/bin/env bash
#$ -N killi
#$ -cwd
#$ -o logs/killi.$JOB_ID.$TASK_ID.out
#$ -e logs/killi.$JOB_ID.$TASK_ID.err
#$ -l h_vmem=16G
#$ -V
#$ -t 1-27
# Annual-killifish water-window experiment.
#
# THE SCENARIO (Dario). Annual killifish live about as long as their pool holds water, while
# time to sexual maturity is conserved across populations. The simple explanation: a
# long-lived ancestor colonises pools that dry sooner, and mutations affecting survival
# BEYOND the water window accumulate because selection can no longer see them. The window
# imposes an ABSOLUTE selection horizon -- unlike Hamilton's gradual decline in the force of
# selection, nothing past it is visible at all.
#
# WHY NO BURN-IN. Every parameter this needs is non-structural, so all arms branch by
# `--override` off the SAME equilibrated ancestor as the three-route experiment
# (burn_s{1,2,3}, AGE_LIMIT=30, MATURATION_AGE=6, K=3000, equilibrated at 430k steps --
# neutral load 0.0908 vs p*=0.0909). Zero burn-in cost.
#
# THE MECHANISM, verified in code rather than taken from the docstring:
#   - ABIOTIC_HAZARD_SHAPE="instant_fatal" returns hazard 1 every ABIOTIC_HAZARD_PERIOD
#     steps and 0 otherwise (abiotic.py:_instant_fatal).
#   - FRAILTY_MODIFIER defaults to 0, so frailty.modify() passes the hazard through
#     undistorted; rng.random() < 1 is always true => a deterministic TOTAL kill.
#   - "abiotic" is in the default MORTALITY_ORDER.
#   - mortality_abiotic() operates on self.population ONLY and never touches self.eggs,
#     so the egg bank survives the dry-down. This is the whole trick.
#   - The abiotic docstring names the target phenomenon outright: "periodic environmental
#     phenomena such as water availability".
#
# THE ANNUAL CYCLE comes from the step order (mortality is step 2, hatch is step 7):
# at step W the pool dries and every fish dies -> no reproduction (nobody alive) ->
# INCUBATION_PERIOD=-1 sees an empty population and the whole egg bank hatches in that
# same step. Synchronous cohort, discrete generations, egg bank in the sediment.
#
# ⚠️ DO NOT SET CARRYING_CAPACITY_EGGS. Its cull takes the LAST N eggs
# (bioreactor.py:323, marked "# TODO biased"). Eggs here accumulate for a whole season
# before hatching, so that would hand a selective advantage to late-season reproduction --
# precisely the axis under test. Left unset, the population is capped instead at hatch time
# by REPRODUCTION_REGULATION, which uses rng.choice(..., replace=False): a genuine random
# sample. Same protection, no bias.
#
# THE SHADOW is ages [W, AGE_LIMIT=30]:
#     W=12 (2x maturation) -> 60% of the age range unseen by selection
#     W=18 (3x)            -> 40%
#     W=24 (4x)            -> 20%
#     W=30 (5x)            -> 0%; coincides with AGE_LIMIT, so it is the natural control
#
# THE PREDICTION. The evolved survival curve should show a KNEE AT W -- survival held
# before it, eroded after -- and the knee should MOVE WITH W across arms. That is the
# environmental analogue of the umbral horizon a*, set by ecology instead of by Ne, and a
# far sharper readout than a gradual selection gradient gives.
#
#   source ~/aegis/routes.env          # CONFIG_DIR=/wins/vlzno/projects/aegis_routes
#   CONFIG_DIR=$CONFIG_DIR TOTAL=630000 qsub -t 1-27 experiments/ne_lifespan/killifish_qsub.sh
#
# Arms share CONFIG_DIR with the three-route experiment because they share its ancestor;
# arm names are prefixed W/ctrl so the two cannot collide.

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

TASK_ID="${SGE_TASK_ID:-1}"
CONFIG_DIR="${CONFIG_DIR:-/wins/vlzno/projects/aegis_routes}"
TOTAL="${TOTAL:-630000}"   # TOTAL steps incl. the 430k burn-in; --extend is not an increment

# Generation mode is INCUBATION_PERIOD:
#   -1  the whole egg bank hatches only once the pool is empty -> one synchronous cohort per
#       season -> NON-OVERLAPPING generations (the annual killifish).
#    3  eggs hatch in waves through the season -> age classes coexist -> OVERLAPPING.
#       Eggs laid since the last wave are still eggs at the dry-down and so survive it;
#       hatch runs after mortality, so a wave landing on the kill step is safe either way.
# LATTICE_MODE=false everywhere: a pool is well mixed, and the ancestor's spatial structure
# is not the variable under test here.
ARM_NAMES=(
    K_ctrl
    K_W12_annual K_W18_annual K_W24_annual K_W30_annual
    K_W12_overlap K_W18_overlap K_W24_overlap K_W30_overlap
)
ARM_OVERRIDE=(
    "LATTICE_MODE=false"
    "LATTICE_MODE=false --override ABIOTIC_HAZARD_SHAPE=instant_fatal --override ABIOTIC_HAZARD_PERIOD=12 --override INCUBATION_PERIOD=-1"
    "LATTICE_MODE=false --override ABIOTIC_HAZARD_SHAPE=instant_fatal --override ABIOTIC_HAZARD_PERIOD=18 --override INCUBATION_PERIOD=-1"
    "LATTICE_MODE=false --override ABIOTIC_HAZARD_SHAPE=instant_fatal --override ABIOTIC_HAZARD_PERIOD=24 --override INCUBATION_PERIOD=-1"
    "LATTICE_MODE=false --override ABIOTIC_HAZARD_SHAPE=instant_fatal --override ABIOTIC_HAZARD_PERIOD=30 --override INCUBATION_PERIOD=-1"
    "LATTICE_MODE=false --override ABIOTIC_HAZARD_SHAPE=instant_fatal --override ABIOTIC_HAZARD_PERIOD=12 --override INCUBATION_PERIOD=3"
    "LATTICE_MODE=false --override ABIOTIC_HAZARD_SHAPE=instant_fatal --override ABIOTIC_HAZARD_PERIOD=18 --override INCUBATION_PERIOD=3"
    "LATTICE_MODE=false --override ABIOTIC_HAZARD_SHAPE=instant_fatal --override ABIOTIC_HAZARD_PERIOD=24 --override INCUBATION_PERIOD=3"
    "LATTICE_MODE=false --override ABIOTIC_HAZARD_SHAPE=instant_fatal --override ABIOTIC_HAZARD_PERIOD=30 --override INCUBATION_PERIOD=3"
)

N_ARMS=${#ARM_NAMES[@]}
SEED_IDX=$(( (TASK_ID - 1) / N_ARMS ))
ARM=$(( (TASK_ID - 1) % N_ARMS ))
ARM_NAME=${ARM_NAMES[$ARM]}
OVERRIDE=${ARM_OVERRIDE[$ARM]}

echo "host: $(hostname) | task: ${TASK_ID} | started: $(date)"

# STRICT glob: match burn_s<N>.yml ONLY. Phase 2 writes an arm config beside every
# run (burn_s1_ctrl.yml, burn_s1_A_ld0000.yml, ...), so a bare burn_s*.yml glob
# silently starts resolving ARM configs as ancestors once any arm has run -- which
# is exactly what killed killifish tasks 10-27 on 2026-08-16.
CONFIG=$(ls "${CONFIG_DIR}"/burn_s*.yml 2>/dev/null | grep -E "/burn_s[0-9]+\.yml$" | sed -n "$(( SEED_IDX + 1 ))p")
if [ -z "${CONFIG}" ]; then
    echo "no burn-in config for seed index ${SEED_IDX} -- is -t larger than n_arms*n_seeds?"
    exit 1
fi
NAME=$(basename "${CONFIG}" .yml)
OUTDIR="${CONFIG_DIR}/${NAME}"
if [ ! -f "${OUTDIR}/.phase1_done" ]; then
    echo "ERROR: ${NAME} has not finished its burn-in. This experiment reuses the three-route"
    echo "       ancestor -- run that phase 1 first and CHECK equilibration."
    exit 1
fi

ARMDIR="${CONFIG_DIR}/${NAME}_${ARM_NAME}"
if [ -f "${ARMDIR}/.arm_done" ]; then
    echo "already complete -- skipping"; exit 0
fi
# Real byte copy per arm; NEVER cp -al. Resume opens the output CSVs in APPEND mode, so a
# hard-linked copy would share an inode with the ancestor and every arm would append into
# one file at once. Stagger so 27 arms do not hit the network mount together.
if [ ! -d "${ARMDIR}" ]; then
    sleep $(( (TASK_ID % N_ARMS) * 3 ))
    cp -r "${OUTDIR}" "${ARMDIR}" || exit 1
    rm -f "${ARMDIR}/.phase1_done"
fi
cp "${CONFIG}" "${ARMDIR}.yml"

echo "run: $(basename "${ARMDIR}")  arm=${ARM_NAME}  extend-to=${TOTAL}"
echo "     override: ${OVERRIDE}"
aegis sim -c "${ARMDIR}.yml" -r --extend "${TOTAL}" --override ${OVERRIDE}
STATUS=$?

# An arm whose water window never fired is a silent null: the config would claim a horizon
# the run never had. Every windowed arm must show abiotic deaths.
if [ ${STATUS} -eq 0 ] && [ "${ARM_NAME}" != "K_ctrl" ]; then
    if ! grep -qi "abiotic" "${ARMDIR}"/*.csv 2>/dev/null && \
       [ ! -s "${ARMDIR}/deaths/abiotic.csv" ] 2>/dev/null; then
        echo "NOTE: could not confirm abiotic deaths were recorded for ${ARM_NAME} --"
        echo "      check the cause-of-death output before interpreting this arm."
    fi
fi
[ ${STATUS} -eq 0 ] && [ -f "${ARMDIR}/output_summary.json" ] && touch "${ARMDIR}/.arm_done"

echo "finished: $(date) | exit: ${STATUS}"
exit ${STATUS}
