"""
Profile aegis simulation to identify performance bottlenecks.

Usage:
    python profiler/profile_sim.py

Produces a detailed breakdown of where time is spent in the simulation loop.
"""

import cProfile
import pstats
import io
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import aegis_sim
from aegis_sim.parameterization import parametermanager
from aegis_sim import variables
from aegis_sim.bioreactor import Bioreactor
from aegis_sim.dataclasses.population import Population
from aegis_sim.recording import recordingmanager
import pathlib
import tempfile
import shutil


def run_profiled_simulation(n_steps=100, pop_size=1000):
    """Run a simulation with cProfile instrumentation."""

    # Create a minimal config for profiling
    config_dir = tempfile.mkdtemp(prefix="aegis_profile_")
    config_path = os.path.join(config_dir, "profile_config.yml")

    with open(config_path, "w") as f:
        f.write(f"""STEPS_PER_SIMULATION: {n_steps}
INITIAL_POPULATION_SIZE: {pop_size}
LOGGING_RATE: 100000
SNAPSHOT_RATE: 100000
PICKLE_RATE: 100000
CHECKPOINT_RATE: 100000
INTERVAL_RATE: 100000
TE_RATE: 100000
TE_DURATION: 1
POPGENSTATS_RATE: 100000
SNAPSHOT_FINAL_COUNT: 0
""")

    config_path = pathlib.Path(config_path).absolute()

    try:
        # Initialize
        aegis_sim.init(config_path, overwrite=True, pickle_path=None, custom_input_params={})
        population = Population.initialize(
            n=parametermanager.parameters.INITIAL_POPULATION_SIZE,
            AGE_LIMIT=parametermanager.parameters.AGE_LIMIT,
        )
        bioreactor = Bioreactor(population)
        bioreactor.eggs = None

        # Warm up numba JIT (don't count compilation time)
        print("Warming up numba JIT...")
        bioreactor.run_step()
        variables.steps += 1

        # Profile the simulation loop
        print(f"\nProfiling {n_steps} steps with population ~{pop_size}...")
        profiler = cProfile.Profile()

        wall_start = time.perf_counter()
        profiler.enable()

        for _ in range(n_steps):
            bioreactor.run_step()
            variables.steps += 1

        profiler.disable()
        wall_end = time.perf_counter()

        wall_time = wall_end - wall_start
        print(f"\nWall time: {wall_time:.2f}s ({wall_time/n_steps*1000:.1f}ms per step)")
        print(f"Final population size: {len(bioreactor.population)}")

        # Print results
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.strip_dirs()

        print("\n" + "=" * 80)
        print("TOP 40 FUNCTIONS BY CUMULATIVE TIME")
        print("=" * 80)
        stats.sort_stats("cumulative")
        stats.print_stats(40)
        print(stream.getvalue())

        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.strip_dirs()

        print("\n" + "=" * 80)
        print("TOP 40 FUNCTIONS BY TOTAL (SELF) TIME")
        print("=" * 80)
        stats.sort_stats("tottime")
        stats.print_stats(40)
        print(stream.getvalue())

        # Also dump to a file for later analysis
        stats_path = os.path.join(config_dir, "profile_stats.prof")
        profiler.dump_stats(stats_path)
        print(f"\nFull profile stats saved to: {stats_path}")
        print("You can analyze with: python -m pstats " + stats_path)

    finally:
        # Cleanup
        try:
            recordingmanager.ticker.stop_process()
        except Exception:
            pass
        shutil.rmtree(config_dir, ignore_errors=True)


if __name__ == "__main__":
    run_profiled_simulation(n_steps=100, pop_size=1000)
