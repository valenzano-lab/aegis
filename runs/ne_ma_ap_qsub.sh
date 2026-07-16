#!/bin/bash
#$ -N ne_ma_ap
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME.$TASK_ID.log
#$ -l h_rt=47:00:00
#$ -l h_vmem=4G
#$ -t 1-36
#
# Ne x {MA, AP} x {sexual, asexual} sweep on Merlin.
#
#   1. python runs/ne_ma_ap_configs.py --outdir configs/ --steps 1000000 --seeds 1 2 3
#   2. mkdir -p logs
#   3. qsub runs/ne_ma_ap_qsub.sh
#
# Adjust -t to match the number of configs generated (36 = 2 arms x 3 Ne x 2 modes x 3 seeds).
#
# IMPORTANT -- wall clock. Measured cost of 1e6 stages:
#     Ne =    300   ~ 8 h
#     Ne =  3,000   ~ 11.5 h
#     Ne = 30,000   ~ 46 h      <-- will hit most wall-clock limits
#
# The Ne=30,000 runs very likely cannot finish in one job. That is fine: configs set
# CHECKPOINT_RATE=100000, so this script RESUMES from the latest checkpoint whenever the
# output directory already exists. If a job is killed at the wall clock, simply qsub the
# same array again -- finished runs exit immediately, unfinished ones pick up where they
# stopped. Repeat until everything reports "already complete".
#
# Scheduler note: written for SGE (qsub / $SGE_TASK_ID). On Slurm use --array and
# $SLURM_ARRAY_TASK_ID; on PBS use $PBS_ARRAYID.

set -u

TASK_ID="${SGE_TASK_ID:-${SLURM_ARRAY_TASK_ID:-${PBS_ARRAYID:-1}}}"

CONFIG=$(ls configs/*.yml | sed -n "${TASK_ID}p")
if [ -z "$CONFIG" ]; then
    echo "No config for task ${TASK_ID}; is -t larger than the number of configs?"
    exit 1
fi

NAME=$(basename "$CONFIG" .yml)
OUTDIR="configs/${NAME}"      # aegis writes output next to the config, named after it

echo "task ${TASK_ID}: ${NAME}"
echo "host: $(hostname)  started: $(date)"

# Already finished? Don't burn a slot re-running it.
# aegis writes output_summary.json only when a run terminates (it carries "runtime"
# and "extinct"), so its presence is a reliable completion marker.
if [ -f "${OUTDIR}/output_summary.json" ]; then
    echo "already complete -- skipping"
    grep -E '"extinct"|"runtime"' "${OUTDIR}/output_summary.json"
    exit 0
fi

if [ -d "${OUTDIR}" ]; then
    echo "output exists -> resuming from latest checkpoint"
    aegis sim -c "$CONFIG" -r
else
    echo "fresh start"
    aegis sim -c "$CONFIG"
fi

STATUS=$?
echo "finished: $(date)  exit: ${STATUS}"
exit ${STATUS}
