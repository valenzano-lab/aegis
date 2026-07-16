#!/usr/bin/env bash
#$ -N nemaap
#$ -cwd
#$ -o logs/nemaap.$TASK_ID.out
#$ -e logs/nemaap.$TASK_ID.err
#$ -l h_vmem=8G
#$ -V
#$ -t 1-36
# SGE array wrapper for the Ne x {MA, AP} x {sexual, asexual} sweep.
#
# gen100 is the HEAD/LOGIN node -- submit from it, never run anything on it.
# Jobs execute on the compute nodes; storage is under /scratch/merlin.
#
#   cd <AEGIS_REPO_ON_CLUSTER>
#   python runs/ne_ma_ap_configs.py --outdir configs/ --steps 1000000 --seeds 1 2 3
#   mkdir -p logs
#   qsub runs/ne_ma_ap_qsub.sh
#   qstat ; cat logs/nemaap.1.out
#
# Set -t to the number of configs generated (36 = 2 arms x 3 Ne x 2 modes x 3 seeds).
#
# ---------------------------------------------------------------------------
# WALL CLOCK -- read this before submitting.
# Measured cost of 1e6 stages (single core):
#     Ne =    300    ~  8 h
#     Ne =  3,000    ~ 11.5 h
#     Ne = 30,000    ~ 46 h
# The 12 Ne=30,000 runs will not finish inside a typical wall-clock limit.
# That is handled: configs set CHECKPOINT_RATE=100000, and this script RESUMES
# (aegis sim -r) whenever an output directory already exists. If jobs are killed,
# just qsub the same array again -- completed runs exit immediately and the rest
# pick up from their latest checkpoint. Repeat until all report "already complete".
#
# No -l h_rt is set, matching the other lab scripts. If Merlin enforces a default
# runtime limit, add one (e.g. #$ -l h_rt=47:00:00) or rely on the resume loop.
# ---------------------------------------------------------------------------
#
# Single-threaded: aegis is numpy-vectorised but not parallel, so no -pe threads.
# If aegis lives in a conda env on Merlin, uncomment and point PATH at it
# (the QTL scripts do this, e.g. .conda/envs/<env>/bin):
# export PATH=/home/<user>/.conda/envs/<aegis_env>/bin:$PATH

set -uo pipefail

TASK_ID="${SGE_TASK_ID:-1}"

CONFIG=$(ls configs/*.yml | sed -n "${TASK_ID}p")
if [ -z "${CONFIG}" ]; then
    echo "no config for task ${TASK_ID} -- is -t larger than the number of configs?"
    exit 1
fi

NAME=$(basename "${CONFIG}" .yml)
OUTDIR="configs/${NAME}"     # aegis writes output beside the config, named after it

echo "host: $(hostname) | task: ${TASK_ID} | run: ${NAME} | started: $(date)"

# aegis writes output_summary.json only when a run terminates (it carries "runtime"
# and "extinct"), so its presence is a reliable completion marker.
if [ -f "${OUTDIR}/output_summary.json" ]; then
    echo "already complete -- skipping"
    grep -E '"extinct"|"runtime"' "${OUTDIR}/output_summary.json"
    exit 0
fi

if [ -d "${OUTDIR}" ]; then
    echo "output exists -> resuming from latest checkpoint"
    aegis sim -c "${CONFIG}" -r
else
    echo "fresh start"
    aegis sim -c "${CONFIG}"
fi
STATUS=$?

echo "finished: $(date) | exit: ${STATUS}"
exit ${STATUS}
