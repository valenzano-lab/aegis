#!/usr/bin/env bash
#$ -N latcal
#$ -cwd
#$ -o logs/latcal.$TASK_ID.out
#$ -e logs/latcal.$TASK_ID.err
#$ -l h_vmem=16G
#$ -V
#$ -t 1-6
# SGE array for the lattice (fragmentation) CALIBRATION -- gate on the Ne->lifespan
# fragmentation arm. Six short runs at IDENTICAL K, differing only in viscosity.
#
# gen100 is the HEAD/LOGIN node -- submit from it, never run anything on it.
#
#   cd ~/aegis            # the cluster checkout (verified)
#   CONFIG_DIR=/wins/vlzno/projects/aegis_latcal
#   python experiments/ne_lifespan/lattice_calibration_configs.py --outdir $CONFIG_DIR
#   mkdir -p logs
#   CONFIG_DIR=$CONFIG_DIR qsub -t 1-6 experiments/ne_lifespan/lattice_calibration_qsub.sh
#   python experiments/ne_lifespan/analyze_lattice_calibration.py $CONFIG_DIR/cal_*/
#
# These are K=3000 x 5000 steps -- minutes to low hours each, not the multi-hour burn-ins
# of the main experiment. Do NOT launch the fragmentation sweep until the analyzer says
# Q1 (N tracks K) and Q2/Q3 (viscosity buys structure range) both pass.
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

# NOT VERIFIED: the env NAME was recorded as an open question and never
# confirmed against a real run. Check `command -v aegis` on gen100 first;
# override with AEGIS_ENV=... before qsub if it is wrong.
AEGIS_ENV="${AEGIS_ENV:-/home/lakatos/dvalenza/.conda/envs/aegis}"
export PATH="${AEGIS_ENV}/bin:${PATH}"

if ! command -v aegis >/dev/null 2>&1; then
    echo "ERROR: 'aegis' not on PATH. AEGIS_ENV=${AEGIS_ENV}"
    exit 1
fi

# The whole calibration is meaningless on an install without the spatial model, and an
# install lacking it would ignore LATTICE_MODE and produce a plausible well-mixed run.
python - <<'PYCHECK' || exit 1
import sys
from aegis_sim.parameterization.default_parameters import DEFAULT_PARAMETERS as D
missing = [p for p in ("LATTICE_MODE", "MIGRATION_RATE", "MIGRATION_LONG_RATE",
                       "LATTICE_TARGET_DENSITY", "LATTICE_RECORD_RATE",
                       "LINEAGE_TRACING") if p not in D]
if missing:
    print("ERROR: this aegis install lacks the spatial model: " + ", ".join(missing))
    sys.exit(1)
print("engine check OK: lattice parameters present")
PYCHECK

TASK_ID="${SGE_TASK_ID:-1}"
CONFIG_DIR="${CONFIG_DIR:-/wins/vlzno/projects/aegis_latcal}"

CONFIG=$(ls "${CONFIG_DIR}"/cal_*.yml 2>/dev/null | sed -n "${TASK_ID}p")
if [ -z "${CONFIG}" ]; then
    echo "no calibration config for task ${TASK_ID} in ${CONFIG_DIR}"
    echo "  generate them: python experiments/ne_lifespan/lattice_calibration_configs.py \\"
    echo "      --outdir ${CONFIG_DIR}"
    exit 1
fi

NAME=$(basename "${CONFIG}" .yml)
OUTDIR="${CONFIG_DIR}/${NAME}"
echo "host: $(hostname) | task: ${TASK_ID} | run: ${NAME} | started: $(date)"

if [ -f "${OUTDIR}/output_summary.json" ]; then
    echo "already complete -- skipping"
    grep -E '"extinct"|"runtime"' "${OUTDIR}/output_summary.json"
    exit 0
fi

# CHECKPOINT_RATE equals STEPS here, so a killed run has no checkpoint to resume from.
# These are short; just start over rather than leaving a half-run directory that aegis
# would refuse to resume on every resubmission.
if [ -d "${OUTDIR}" ]; then
    echo "incomplete output dir -> discarding and restarting"
    rm -rf "${OUTDIR}"
fi

aegis sim -c "${CONFIG}"
STATUS=$?

# The silent failure this calibration exists to rule out: LATTICE_MODE=True but the
# lattice never engaged. Catch it here, at the run, rather than in the analysis.
if [ ${STATUS} -eq 0 ] && grep -q "^LATTICE_MODE: true" "${CONFIG}"; then
    if [ ! -d "${OUTDIR}/lattice" ] || [ -z "$(ls -A "${OUTDIR}/lattice" 2>/dev/null)" ]; then
        echo "ERROR: ${NAME} sets LATTICE_MODE=true but wrote no lattice snapshots."
        echo "       The spatial model did not engage -- do not interpret this run."
        STATUS=1
    fi
fi

echo "finished: $(date) | exit: ${STATUS}"
exit ${STATUS}
