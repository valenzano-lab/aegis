#!/bin/bash
# Waits for both equilibrium pickles then launches hybrid runs.
AEGIS=/Users/dvalenzano/Dropbox/Lab/git/projects/aegis/.venv/bin/aegis
RUNDIR=/Users/dvalenzano/Dropbox/Lab/git/projects/aegis

until [ -f "$RUNDIR/runs/pop1_equilibrium/pickles/50000" ] && \
      [ -f "$RUNDIR/runs/pop2_equilibrium/pickles/50000" ]; do
    sleep 30
done

echo "$(date): Both pickles ready — launching hybrid runs"

cd "$RUNDIR"
"$AEGIS" sim -c runs/hybrid_run_1_2.yml -o > /tmp/hybrid_1_2.log 2>&1 &
H1=$!
"$AEGIS" sim -c runs/hybrid_run_2_1.yml -o > /tmp/hybrid_2_1.log 2>&1 &
H2=$!

echo "hybrid_1_2 PID=$H1  hybrid_2_1 PID=$H2"
wait $H1 $H2
echo "$(date): Both hybrid runs complete"
