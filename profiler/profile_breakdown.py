"""
Detailed timing breakdown of each phase within a simulation step.
"""

import time
import sys
import os
import numpy as np

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
from collections import defaultdict


def run_breakdown(n_steps=200, pop_size=1000):
    config_dir = tempfile.mkdtemp(prefix="aegis_profile_")
    config_path = os.path.join(config_dir, "profile_config.yml")

    with open(config_path, "w") as f:
        f.write(f"""STEPS_PER_SIMULATION: {n_steps + 10}
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
        aegis_sim.init(config_path, overwrite=True, pickle_path=None, custom_input_params={})
        population = Population.initialize(
            n=parametermanager.parameters.INITIAL_POPULATION_SIZE,
            AGE_LIMIT=parametermanager.parameters.AGE_LIMIT,
        )
        bioreactor = Bioreactor(population)
        bioreactor.eggs = None

        # Warm up
        for _ in range(5):
            bioreactor.run_step()
            variables.steps += 1

        # Monkey-patch run_step to time each phase
        timings = defaultdict(list)
        pop_sizes = []

        original_run_step = bioreactor.run_step

        for step_i in range(n_steps):
            pop_sizes.append(len(bioreactor.population))

            if len(bioreactor) == 0:
                break

            # Time mortalities
            t0 = time.perf_counter()
            bioreactor.mortalities()
            t1 = time.perf_counter()
            timings["mortalities"].append(t1 - t0)

            # Time resource replenish
            from aegis_sim.submodels.resources.resources import resources
            t0 = time.perf_counter()
            resources.replenish()
            t1 = time.perf_counter()
            timings["resource_replenish"].append(t1 - t0)

            # Time recording (before reproduction)
            t0 = time.perf_counter()
            recordingmanager.popsizerecorder.write_before_reproduction(bioreactor.population)
            t1 = time.perf_counter()
            timings["recording_popsize"].append(t1 - t0)

            # Time growth
            t0 = time.perf_counter()
            bioreactor.growth()
            t1 = time.perf_counter()
            timings["growth"].append(t1 - t0)

            # Time reproduction
            t0 = time.perf_counter()
            bioreactor.reproduction()
            t1 = time.perf_counter()
            timings["reproduction"].append(t1 - t0)

            # Time aging
            t0 = time.perf_counter()
            bioreactor.age()
            t1 = time.perf_counter()
            timings["aging"].append(t1 - t0)

            # Time hatching
            t0 = time.perf_counter()
            bioreactor.hatch()
            t1 = time.perf_counter()
            timings["hatching"].append(t1 - t0)

            # Time envdrift
            from aegis_sim import submodels
            t0 = time.perf_counter()
            submodels.architect.envdrift.evolve(step=variables.steps)
            t1 = time.perf_counter()
            timings["envdrift"].append(t1 - t0)

            # Time recording (after reproduction + all other recorders)
            t0 = time.perf_counter()
            recordingmanager.popsizerecorder.write_after_reproduction(bioreactor.population)
            recordingmanager.popsizerecorder.write_egg_num_after_reproduction(bioreactor.eggs)
            recordingmanager.envdriftmaprecorder.write(step=variables.steps)
            recordingmanager.flushrecorder.collect("additive_age_structure", bioreactor.population.ages)
            recordingmanager.picklerecorder.write(bioreactor.population)
            recordingmanager.featherrecorder.write(bioreactor.population)
            recordingmanager.guirecorder.record(bioreactor.population)
            recordingmanager.flushrecorder.flush()
            recordingmanager.popgenstatsrecorder.write(
                bioreactor.population.genomes,
                bioreactor.population.phenotypes.extract(ages=bioreactor.population.ages, trait_name="muta"),
            )
            recordingmanager.summaryrecorder.record_memuse()
            recordingmanager.terecorder.record(bioreactor.population.ages, "alive")
            recordingmanager.checkpointrecorder.write(bioreactor.population, bioreactor.eggs)
            t1 = time.perf_counter()
            timings["recording_all"].append(t1 - t0)

            variables.steps += 1

        # Print results
        print(f"\nTiming breakdown over {len(pop_sizes)} steps")
        print(f"Average population size: {np.mean(pop_sizes):.0f}")
        print(f"Min/Max population size: {np.min(pop_sizes)}/{np.max(pop_sizes)}")
        print()

        total = sum(np.sum(v) for v in timings.values())
        print(f"{'Phase':<25} {'Total (s)':>10} {'Avg (ms)':>10} {'% of total':>10}")
        print("-" * 60)

        sorted_timings = sorted(timings.items(), key=lambda x: -np.sum(x[1]))
        for name, times in sorted_timings:
            t_total = np.sum(times)
            t_avg = np.mean(times) * 1000
            pct = t_total / total * 100
            print(f"{name:<25} {t_total:>10.3f} {t_avg:>10.2f} {pct:>9.1f}%")

        print("-" * 60)
        print(f"{'TOTAL':<25} {total:>10.3f} {total/len(pop_sizes)*1000:>10.2f} {'100.0':>9}%")

    finally:
        try:
            recordingmanager.ticker.stop_process()
        except Exception:
            pass
        shutil.rmtree(config_dir, ignore_errors=True)


if __name__ == "__main__":
    run_breakdown(n_steps=200, pop_size=1000)
