#!/usr/bin/env bash
#$ -N nelife
#$ -cwd
#$ -o logs/nelife$PHASE.$TASK_ID.out
#$ -e logs/nelife$PHASE.$TASK_ID.err
#$ -l h_vmem=16G
#$ -V
#$ -t 1-3
# SGE array wrapper for the Ne -> lifespan experiment (full range Ne 1e2..1e4).
#
# gen100 is the HEAD/LOGIN node -- submit from it, never run anything on it.
#
# THE QUESTION (Ruchitha): is the generational lifespan reduction driven by carrying
# capacity K or by Ne? THE DESIGN (Dario): pre-evolve ONE population, subsample it to a
# range of Ne, run each forward at CONSTANT K=N (birth-regulated, no starvation), then
# re-measure realized Ne and relate it to evolved lifespan. A common ancestor is what
# makes forward divergence attributable to Ne alone.
#
#   PHASE 1  burn-in at K_BURN, one job per seed.
#   PHASE 2  per (N, seed): subsample the seed's burn-in pickle, then run forward at K=N.
#
# ---------------------------------------------------------------------------
# HOW TO RUN
#
#   cd ~/aegis            # the cluster checkout (verified)
#   CONFIG_DIR=/wins/vlzno/projects/aegis_ne_lifespan
#   python experiments/ne_lifespan/ne_lifespan_configs.py --outdir $CONFIG_DIR
#   mkdir -p logs
#
#   CONFIG_DIR=$CONFIG_DIR PHASE=1 qsub -t 1-3  experiments/ne_lifespan/ne_lifespan_qsub.sh
#   # WAIT for phase 1, then CHECK the burn-ins before spending phase-2 compute:
#   #   - final population size ~= K_BURN (a crash invalidates every arm off that seed)
#   #   - python runs/check_equilibration.py $CONFIG_DIR/burn_s1
#   CONFIG_DIR=$CONFIG_DIR PHASE=2 qsub -t 1-15 experiments/ne_lifespan/ne_lifespan_qsub.sh
#
#   # analysis (needs the aegis env, or any env with pandas+pyarrow):
#   python experiments/ne_lifespan/analyze_contrast.py $CONFIG_DIR/fwd_N*_s1
#   python runs/genetic_ne.py $CONFIG_DIR/fwd_N*_s*        # realized Ne, units-correct
#
# -t must match the config count: phase 1 = n_seeds, phase 2 = n_seeds * n_targets.
# ne_lifespan_configs.py prints both numbers when it writes the configs.
#
# PHASE must be EXPORTED at submit time (as above), not merely set: both phases use the
# same task numbers, and #$ -V is what carries PHASE into the log filename.
# ---------------------------------------------------------------------------
# WALL CLOCK. The burn-in at K=10000 is the heavy part -- by the ne_ma_ap measurements
# (~11.5 h for 1e6 steps at Ne=3,000, ~46 h at Ne=30,000) expect a 200k-step burn-in at
# K=10000 to run several hours. Forward arms are cheap at the low-Ne end and comparable
# to the burn-in at N=10000. No -l h_rt is set, matching the other lab scripts; both
# phases resume from CHECKPOINT_RATE=10000, so if jobs are killed just qsub the same
# array again -- completed runs exit immediately and the rest pick up where they stopped.
#
# NOTE this experiment does NOT need the cp -r per arm that oscillation_qsub.sh does.
# There, every arm RESUMED a shared burnt-in directory, so each needed its own byte copy
# (resume appends to the output CSVs in place). Here each arm starts FRESH from a
# subsampled pickle into its own output directory -- nothing is shared but the pickle,
# which is read-only. Much cheaper on the network mount.
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

# VERIFIED 2026-08-15 on gen100: this env exists and holds bin/aegis, and its
# aegis_sim is an EDITABLE install pointing at ~/aegis -- so checking out a branch
# in that repo updates the engine, no pip install needed. `conda activate aegis`
# before qsub anyway: #$ -V then carries it to the compute nodes as well.
AEGIS_ENV="${AEGIS_ENV:-/home/lakatos/dvalenza/.conda/envs/aegis}"
export PATH="${AEGIS_ENV}/bin:${PATH}"

if ! command -v aegis >/dev/null 2>&1; then
    echo "ERROR: 'aegis' not on PATH. AEGIS_ENV=${AEGIS_ENV}"
    exit 1
fi

# The forward arms depend on seed mode (aegis sim -p PICKLE) and on parameters the
# pilot architecture sets. An install missing either would run a DIFFERENT experiment
# and still exit 0, so fail loudly instead.
python - <<'PYCHECK' || exit 1
import sys
from aegis_sim.parameterization.default_parameters import DEFAULT_PARAMETERS as D
from aegis.parse import get_parser
missing = [p for p in ("MAX_OFFSPRING_NUMBER", "STARVATION_PENALTY",
                       "REPRODUCTION_REGULATION", "POPGENSTATS_SAMPLE_SIZE") if p not in D]
try:
    ns = get_parser().parse_args(["sim", "-c", "probe.yml", "-p", "probe.pkl"])
    ok = getattr(ns, "pickle_path", None) == "probe.pkl"
except SystemExit:
    ok = False
if not ok:
    missing.append("'aegis sim -p' seed mode")
if missing:
    print("ERROR: this aegis install is out of date, missing: " + ", ".join(missing))
    sys.exit(1)
print("engine check OK: pilot parameters + seed mode present")
PYCHECK

TASK_ID="${SGE_TASK_ID:-1}"
PHASE="${PHASE:-1}"
# Configs live here and so does output -- aegis writes beside the config, named after
# its stem. Keep it OUTSIDE the git tree: the sweep produces GBs.
CONFIG_DIR="${CONFIG_DIR:-/wins/vlzno/projects/aegis_ne_lifespan}"

echo "host: $(hostname) | phase: ${PHASE} | task: ${TASK_ID} | started: $(date)"

# Run a config, resuming when possible. A job killed before the first checkpoint leaves
# an output directory with no checkpoint file, which aegis refuses to resume -- clear it
# and start over rather than dying on every resubmission. $2, if given, is a seed pickle.
run_config() {
    local config="$1" seed_pickle="${2:-}"
    local name outdir
    name=$(basename "${config}" .yml)
    outdir="${CONFIG_DIR}/${name}"

    if [ -f "${outdir}/output_summary.json" ]; then
        echo "already complete -- skipping"
        grep -E '"extinct"|"runtime"' "${outdir}/output_summary.json"
        return 0
    fi

    if [ -d "${outdir}" ] && [ ! -e "${outdir}/checkpoint" ]; then
        echo "output dir exists but has no checkpoint (killed before step CHECKPOINT_RATE)"
        echo "  -> discarding ${outdir} and starting over"
        rm -rf "${outdir}"
    fi

    if [ -d "${outdir}" ]; then
        echo "resuming from latest checkpoint"
        aegis sim -c "${config}" -r
    elif [ -n "${seed_pickle}" ]; then
        echo "fresh start, seeded from ${seed_pickle}"
        aegis sim -c "${config}" -p "${seed_pickle}"
    else
        echo "fresh start"
        aegis sim -c "${config}"
    fi
}

if [ "${PHASE}" = "1" ]; then
    # ---- BURN-IN: one common ancestor per seed --------------------------------
    CONFIG=$(ls "${CONFIG_DIR}"/burn_s*.yml 2>/dev/null | sed -n "${TASK_ID}p")
    if [ -z "${CONFIG}" ]; then
        echo "no burn-in config for task ${TASK_ID} in ${CONFIG_DIR}"
        echo "  generate them: python experiments/ne_lifespan/ne_lifespan_configs.py \\"
        echo "      --outdir ${CONFIG_DIR}"
        exit 1
    fi
    NAME=$(basename "${CONFIG}" .yml)
    OUTDIR="${CONFIG_DIR}/${NAME}"
    echo "run: ${NAME} (burn-in)"

    # output_summary.json alone cannot mark phase 1 done -- nothing here resumes the
    # burn-in directory, but keep the explicit marker so phase 2's gate is unambiguous.
    if [ -f "${OUTDIR}/.phase1_done" ]; then
        echo "already complete -- skipping"
        exit 0
    fi
    run_config "${CONFIG}"
    STATUS=$?
    [ ${STATUS} -eq 0 ] && [ -f "${OUTDIR}/output_summary.json" ] && touch "${OUTDIR}/.phase1_done"

else
    # ---- FORWARD: subsample the ancestor to N, run at constant K=N -------------
    # One task per fwd_N{N}_s{seed}.yml, in ls order -- the grid lives in the config
    # generator, not duplicated here, so the two cannot drift apart.
    CONFIG=$(ls "${CONFIG_DIR}"/fwd_N*_s*.yml 2>/dev/null | sed -n "${TASK_ID}p")
    if [ -z "${CONFIG}" ]; then
        echo "no forward config for task ${TASK_ID} in ${CONFIG_DIR}"
        echo "  -- is -t larger than n_seeds * n_targets?"
        exit 1
    fi
    NAME=$(basename "${CONFIG}" .yml)     # fwd_N00316_s2
    SEED="${NAME##*_s}"                   # 2
    N="${NAME#fwd_N}"; N="${N%%_*}"       # 00316 (zero-padded so the array sorts by N)
    N=$((10#$N))                          # 316 -- the 10# is REQUIRED, bash reads a
                                          # leading-zero literal as OCTAL otherwise
    BURNDIR="${CONFIG_DIR}/burn_s${SEED}"
    SUBPKL="${CONFIG_DIR}/sub_${NAME#fwd_}.pkl"   # sub_N00316_s2.pkl, one per arm

    echo "run: ${NAME}  N=K=${N}  seed=${SEED}"

    if [ ! -f "${BURNDIR}/.phase1_done" ]; then
        echo "ERROR: burn_s${SEED} has not finished phase 1 -- run PHASE=1 first, and CHECK"
        echo "       the burn-in reached K and equilibrated before spending phase-2 compute."
        exit 1
    fi

    # Highest-numbered pickle = the final step. PickleRecorder always writes on the last
    # step regardless of PICKLE_RATE, so this is the fully burnt-in ancestor.
    BURN_PICKLE=$(ls "${BURNDIR}/pickles" 2>/dev/null | sort -n | tail -1)
    if [ -z "${BURN_PICKLE}" ]; then
        echo "ERROR: no pickles in ${BURNDIR}/pickles -- burn-in produced no seed population."
        exit 1
    fi
    BURN_PICKLE="${BURNDIR}/pickles/${BURN_PICKLE}"

    # Each task writes its OWN subsample file, so concurrent arms never race; the burn-in
    # pickle they all read is untouched. Deterministic in (seed, N): a resubmitted task
    # reproduces its own draw.
    if [ ! -f "${SUBPKL}" ]; then
        python experiments/ne_lifespan/subsample.py \
            "${BURN_PICKLE}" "${N}" "${SUBPKL}" --rng-seed "${SEED}" || exit 1
    else
        echo "subsample already present: ${SUBPKL}"
    fi

    run_config "${CONFIG}" "${SUBPKL}"
    STATUS=$?
fi

echo "finished: $(date) | exit: ${STATUS}"
exit ${STATUS}
