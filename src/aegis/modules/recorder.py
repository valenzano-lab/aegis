"""Data recorder

Records data generated by the simulation.

When thinking about recording additional data, consider that there are three recording methods:
    I. Snapshots (record data from the population at a specific stage)
    II. Flushes (collect data over time then flush)
    III. One-time records
"""

import pandas as pd
import numpy as np
import json
import time
import copy
import psutil
import subprocess
import logging

from aegis.pan import cnf
from aegis import pan
from aegis.pan import var
from aegis.modules.popgenstats import PopgenStats
from aegis.help.config import causeofdeath_valid
from aegis.modules import genetics


def get_dhm(timediff):
    """Format time in a human-readable format."""
    d = int(timediff / 86400)
    timediff %= 86400
    h = int(timediff / 3600)
    timediff %= 3600
    m = int(timediff / 60)
    return f"{d}`{h:02}:{m:02}"


def _log_progress(popsize="?"):
    """Record some information about the time and speed of simulation."""

    if pan.skip(cnf.LOGGING_RATE):
        return

    logging.info("%8s / %s / N=%s", var.stage, cnf.STAGES_PER_SIMULATION, popsize)

    # Get time estimations
    time_diff = time.time() - pan.time_start

    seconds_per_100 = time_diff / var.stage * 100
    eta = (cnf.STAGES_PER_SIMULATION - var.stage) / 100 * seconds_per_100

    stages_per_min = int(var.stage / (time_diff / 60))

    runtime = get_dhm(time_diff)
    time_per_1M = get_dhm(time_diff / var.stage * 1000000)
    eta = get_dhm(eta)

    # Save time estimations
    content = (var.stage, eta, time_per_1M, runtime, stages_per_min, popsize)
    with open(pan.progress_path, "ab") as f:
        np.savetxt(f, [content], fmt="%-10s", delimiter="| ")


print(pan.output_path)
opath = pan.output_path
paths = {
    "BASE_DIR": opath,
    "snapshots_genotypes": opath / "snapshots" / "genotypes",
    "snapshots_phenotypes": opath / "snapshots" / "phenotypes",
    "snapshots_demography": opath / "snapshots" / "demography",
    "visor": opath / "visor",
    "visor_spectra": opath / "visor" / "spectra",
    "input_summary": opath,
    "output_summary": opath,
    "pickles": opath / "pickles",
    "popgen": opath / "popgen",
    "phenomap": opath,
    "te": opath / "te",
}
for path in paths.values():
    path.mkdir(exist_ok=True, parents=True)

# Initialize collection
_collection = {
    "age_at_birth": [0] * cnf.MAX_LIFESPAN,
    "additive_age_structure": [0] * cnf.MAX_LIFESPAN,
}

_collection.update({f"age_at_{causeofdeath}": [0] * cnf.MAX_LIFESPAN for causeofdeath in causeofdeath_valid})

collection = copy.deepcopy(_collection)

# Needed for output summary
extinct = False

# Memory utilization
memory_use = []
psutil_process = psutil.Process()

# PopgenStats
popgenstats = PopgenStats()

# other
te_record_number = 0

# Add headers
for key in _collection.keys():
    with open(paths["visor_spectra"] / f"{key}.csv", "ab") as f:
        array = np.arange(cnf.MAX_LIFESPAN)
        np.savetxt(f, [array], delimiter=",", fmt="%i")

with open(paths["visor"] / "genotypes.csv", "ab") as f:
    array = np.arange(genetics.get_number_of_bits())  # (ploidy, length, bits_per_locus)
    np.savetxt(f, [array], delimiter=",", fmt="%i")

with open(paths["visor"] / "phenotypes.csv", "ab") as f:
    array = np.arange(genetics.get_number_of_phenotypic_values())  # number of phenotypic values
    np.savetxt(f, [array], delimiter=",", fmt="%i")

# ===============================
# RECORDING METHOD I. (snapshots)
# ===============================


def record_memory_use():
    # TODO refine
    memory_use_ = psutil_process.memory_info()[0] / float(2**20)
    memory_use.append(memory_use_)
    if len(memory_use) > 1000:
        memory_use.pop(0)


def record_visor(population):
    """Record data that is needed by visor."""
    if pan.skip(cnf.VISOR_RATE) or len(population) == 0:
        return

    # genotypes.csv | Record allele frequency
    with open(paths["visor"] / "genotypes.csv", "ab") as f:
        array = population.genomes.reshape(len(population), -1).mean(0)
        np.savetxt(f, [array], delimiter=",", fmt="%1.3e")

    # phenotypes.csv | Record median phenotype
    with open(paths["visor"] / "phenotypes.csv", "ab") as f:
        array = np.median(population.phenotypes, 0)
        np.savetxt(f, [array], delimiter=",", fmt="%1.3e")

    flush()


def record_snapshots(population):
    """Record demographic, genetic and phenotypic data from the current population."""
    if pan.skip(cnf.SNAPSHOT_RATE) or len(population) == 0:
        return

    logging.debug(f"Snapshots recorded at stage {var.stage}")

    # genotypes
    df_gen = pd.DataFrame(np.array(population.genomes.reshape(len(population), -1)))
    df_gen.reset_index(drop=True, inplace=True)
    df_gen.columns = [str(c) for c in df_gen.columns]
    df_gen.to_feather(paths["snapshots_genotypes"] / f"{var.stage}.feather")

    # phenotypes
    df_phe = pd.DataFrame(np.array(population.phenotypes))
    df_phe.reset_index(drop=True, inplace=True)
    df_phe.columns = [str(c) for c in df_phe.columns]
    df_phe.to_feather(paths["snapshots_phenotypes"] / f"{var.stage}.feather")

    # demography
    dem_attrs = ["ages", "births", "birthdays"]
    demo = {attr: getattr(population, attr) for attr in dem_attrs}
    df_dem = pd.DataFrame(demo, columns=dem_attrs)
    df_dem.reset_index(drop=True, inplace=True)
    df_dem.to_feather(paths["snapshots_demography"] / f"{var.stage}.feather")


def record_popgenstats(genomes, mutation_rates):
    """Record population size in popgenstats, and record popgen statistics."""
    popgenstats.record_pop_size_history(genomes)

    if pan.skip(cnf.POPGENSTATS_RATE) or len(genomes) == 0:
        return

    popgenstats.calc(genomes, mutation_rates)

    # Record simple statistics
    array = list(popgenstats.emit_simple().values())
    if None in array:
        return

    with open(paths["popgen"] / "simple.csv", "ab") as f:
        np.savetxt(f, [array], delimiter=",", fmt="%1.3e")

    # Record complex statistics
    complex_statistics = popgenstats.emit_complex()
    for key, array in complex_statistics.items():
        with open(paths["popgen"] / f"{key}.csv", "ab") as f:
            np.savetxt(f, [array], delimiter=",", fmt="%1.3e")


def record_pickle(population):
    if pan.skip(cnf.PICKLE_RATE) and not var.stage == 1:  # Also records the pickle before the first stage
        return

    logging.debug(f"pickle recorded at stage {var.stage}")

    pickle_path = paths["pickles"] / str(var.stage)
    population.save_pickle_to(pickle_path)


# ==============================
# RECORDING METHOD II. (flushes)
# ==============================


def collect(key, ages):
    """Add data into memory which will be recorded later."""
    collection[key] += np.bincount(ages, minlength=cnf.MAX_LIFESPAN)


def flush():
    """Record data that has been collected over time."""
    # spectra/*.csv | Age distribution of various subpopulations (e.g. population that died of genetic causes)

    global collection

    for key, val in collection.items():
        with open(paths["visor_spectra"] / f"{key}.csv", "ab") as f:
            array = np.array(val)
            np.savetxt(f, [array], delimiter=",", fmt="%i")

    # Reinitialize the collection
    collection = copy.deepcopy(_collection)


# =================================
# RECORDING METHOD III. (record once)
# =================================


def record_phenomap(map_):
    with open(paths["phenomap"] / "phenomap.csv", "w") as f:
        np.savetxt(f, map_, delimiter=",", fmt="%f")


@staticmethod
def get_folder_size_with_du(folder_path):
    result = subprocess.run(["du", "-sh", folder_path], stdout=subprocess.PIPE, text=True)
    return result.stdout.split()[0]


def record_output_summary():
    try:
        storage_use = get_folder_size_with_du(pan.output_path)
    except:
        storage_use = ""

    summary = {
        "extinct": extinct,
        "random_seed": var.random_seed,
        "time_start": pan.time_start,
        "runtime": time.time() - pan.time_start,
        "jupyter_path": str(pan.output_path.absolute()),
        "memory_use": np.median(memory_use),
        "storage_use": storage_use,
    }
    with open(paths["output_summary"] / "output_summary.json", "w") as f:
        json.dump(summary, f, indent=4)


def record_input_summary():
    summary = {
        # "extinct": extinct,
        "random_seed": var.random_seed,
        "time_start": pan.time_start,
        # "time_end": time.time(),
        "jupyter_path": str(pan.output_path.absolute()),
    }
    with open(paths["input_summary"] / "input_summary.json", "w") as f:
        json.dump(summary, f, indent=4)


# =================================
# RECORDING METHOD IV. (other)
# =================================


def record_TE(T, e):
    """
    Record deaths.
    T .. time / duration (ages)
    E .. event observed (0/alive or 1/dead)

    ###

    To fit this data using lifelines, use this script as inspiration:
        from lifelines import KaplanMeierFitter
        kmf = KaplanMeierFitter()
        te = pd.read_csv("/path/to/te/1.csv")
        kmf.fit(te["T"], te["E"])
        kmf.survival_function_.plot()

    You can compare this to observed survivorship curves:
        analyzer.get_total_survivorship(container).plot()

    """

    global te_record_number

    assert e in ("alive", "dead")

    if (var.stage % cnf.TE_RATE) == 0 or var.stage == 1:
        # open new file and add header
        with open(paths["te"] / f"{te_record_number}.csv", "w") as file_:
            array = ["T", "E"]
            np.savetxt(file_, [array], delimiter=",", fmt="%s")

    elif ((var.stage % cnf.TE_RATE) < cnf.TE_DURATION) and e == "dead":
        # record deaths
        E = np.repeat(1, len(T))
        data = np.array([T, E]).T
        with open(paths["te"] / f"{te_record_number}.csv", "ab") as file_:
            np.savetxt(file_, data, delimiter=",", fmt="%i")

    elif (((var.stage % cnf.TE_RATE) == cnf.TE_DURATION) or var.stage == cnf.STAGES_PER_SIMULATION) and e == "alive":
        # flush
        logging.debug(f"Data for survival analysis (T,E) flushed at stage {var.stage}")
        E = np.repeat(0, len(T))
        data = np.array([T, E]).T
        with open(paths["te"] / f"{te_record_number}.csv", "ab") as file_:
            np.savetxt(file_, data, delimiter=",", fmt="%i")

        te_record_number += 1


phenomap_ = genetics.get_map()
if phenomap_ is not None:
    record_phenomap(genetics.get_map())
