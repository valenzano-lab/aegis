import collections
import csv
import json
import logging
import pathlib

import numpy as np

from aegis_sim import variables, parameterization
from aegis_sim.parameterization import parametermanager
from aegis_sim.utilities.funcs import skip
from .recorder import Recorder


class EquilibriumRecorder(Recorder):
    """
    Periodically checks whether the population has reached an evolutionary
    equilibrium (population size and phenotype trait-age medians stop
    trending). Every EQUILIBRIUM_CHECK_RATE steps, the current stability
    metrics are appended to equilibrium_progress.csv regardless of whether
    equilibrium has been reached yet.

    If EQUILIBRIUM_TERMINATION is enabled and equilibrium is detected, the
    simulation is NOT stopped on the spot. Instead, STEPS_PER_SIMULATION is
    shortened to end SNAPSHOT_FINAL_COUNT steps later (the same "final
    steps" window AEGIS already uses to take extra end-of-simulation
    snapshots -- see featherrecorder.py), so the run winds down through the
    normal end-of-simulation path: final snapshots are recorded, all
    recorders flush and close cleanly, and output_summary.json is written
    exactly as it would be for a full run, just shorter. The event is
    logged immediately (with the metric values) and written to
    equilibrium_summary.json.

    Stability is judged separately across three phenotype column groups
    (in addition to population size):
      - overall: every evolvable, age-specific trait-age column
      - immature: ages below MATURATION_AGE
      - early_mature: ages from MATURATION_AGE up to the midpoint between
        MATURATION_AGE and the end of the reproductive age range
        (REPRODUCTION_ENDPOINT if set, else AGE_LIMIT)
    """

    def __init__(self, odir: pathlib.Path):
        self.odir = odir
        self.enabled = parametermanager.parameters.EQUILIBRIUM_TERMINATION
        self.window = parametermanager.parameters.EQUILIBRIUM_WINDOW
        self.tolerance = parametermanager.parameters.EQUILIBRIUM_TOLERANCE
        self.popsize_history = collections.deque(maxlen=self.window)
        self.phenotype_history = collections.deque(maxlen=self.window)
        self.equilibrium_reached = False

        if self.enabled:
            self.original_steps_per_simulation = parametermanager.parameters.STEPS_PER_SIMULATION
            self.maturation_age = parametermanager.parameters.MATURATION_AGE
            mature_age_end = parametermanager.parameters.REPRODUCTION_ENDPOINT or parametermanager.parameters.AGE_LIMIT
            self.mid_age = self.maturation_age + (mature_age_end - self.maturation_age) / 2
            self.immature_columns, self.early_mature_columns = self._build_age_band_columns()
            self._init_progress_file()

    def _build_age_band_columns(self):
        """Map phenotype-vector column indices to (trait, age) and bucket them by age band.
        Only evolvable, age-specific traits carry a per-age column; other traits are skipped."""
        immature_cols = []
        early_mature_cols = []
        for trait in parameterization.traits.values():
            if not trait.evolvable or trait.agespecific is not True:
                continue
            for age in range(trait.length):
                col = trait.start + age
                if age < self.maturation_age:
                    immature_cols.append(col)
                elif self.maturation_age <= age <= self.mid_age:
                    early_mature_cols.append(col)
        return np.array(immature_cols, dtype=int), np.array(early_mature_cols, dtype=int)

    def _init_progress_file(self):
        with open(self.odir / "equilibrium_progress.csv", "w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "step",
                    "popsize",
                    "popsize_cv",
                    "phenotype_cv_overall",
                    "phenotype_cv_immature",
                    "phenotype_cv_early_mature",
                    "n_checks_so_far",
                    "equilibrium_reached",
                ]
            )

    def check(self, population):
        """Call once per step; internally rate-limited by EQUILIBRIUM_CHECK_RATE."""
        if not self.enabled or self.equilibrium_reached:
            return

        if skip("EQUILIBRIUM_CHECK_RATE") or len(population) == 0:
            return

        self.popsize_history.append(len(population))
        self.phenotype_history.append(np.median(population.phenotypes.get(), 0))

        if len(self.popsize_history) < self.window:
            metrics = None
            is_stable = False
        else:
            metrics = self._stability_metrics()
            is_stable = all(cv < self.tolerance for cv in metrics.values())

        self._write_progress_row(metrics=metrics, is_stable=is_stable)

        if is_stable:
            self._trigger_equilibrium(metrics)

    def _trigger_equilibrium(self, metrics: dict):
        self.equilibrium_reached = True

        # Schedule a clean wind-down instead of stopping on the spot: reuse SNAPSHOT_FINAL_COUNT,
        # the same "final steps" window AEGIS already uses for end-of-simulation snapshots.
        grace_period = parametermanager.parameters.SNAPSHOT_FINAL_COUNT
        new_end = min(variables.steps + grace_period, self.original_steps_per_simulation)
        steps_saved = self.original_steps_per_simulation - new_end
        parametermanager.parameters.STEPS_PER_SIMULATION = new_end

        self._write_summary(metrics=metrics, new_end=new_end, steps_saved=steps_saved)

        logging.info(
            f"EQUILIBRIUM REACHED at step {variables.steps}: popsize={self.popsize_history[-1]} "
            f"(popsize_cv={metrics['popsize_cv']:.4f}), "
            f"phenotype_cv_overall={metrics['phenotype_cv_overall']:.4f}, "
            f"phenotype_cv_immature={metrics['phenotype_cv_immature']:.4f}, "
            f"phenotype_cv_early_mature={metrics['phenotype_cv_early_mature']:.4f} "
            f"(tolerance={self.tolerance:.4f}, stable over the last {self.window} checks, "
            f"{parametermanager.parameters.EQUILIBRIUM_CHECK_RATE} steps apart). "
            f"Winding down: STEPS_PER_SIMULATION shortened from {self.original_steps_per_simulation} to "
            f"{new_end} ({steps_saved} steps saved), so the run still finishes through the normal "
            f"end-of-simulation path (final snapshots, clean recorder shutdown)."
        )

    def _stability_metrics(self) -> dict:
        popsize_arr = np.array(self.popsize_history, dtype=float)
        phenotype_arr = np.array(self.phenotype_history)  # shape: (window, n_trait_age_columns)
        all_columns = np.arange(phenotype_arr.shape[1])
        return {
            "popsize_cv": self._coefficient_of_variation(popsize_arr),
            "phenotype_cv_overall": self._cv_for_columns(phenotype_arr, all_columns),
            "phenotype_cv_immature": self._cv_for_columns(phenotype_arr, self.immature_columns),
            "phenotype_cv_early_mature": self._cv_for_columns(phenotype_arr, self.early_mature_columns),
        }

    @staticmethod
    def _cv_for_columns(phenotype_arr: np.ndarray, columns: np.ndarray) -> float:
        if len(columns) == 0:
            return 0.0
        sub = phenotype_arr[:, columns]
        means = sub.mean(axis=0)
        stds = sub.std(axis=0)
        # Ignore columns with a negligible mean (e.g. a trait pinned near 0) to avoid a noisy divide-by-near-zero
        active = means > 1e-9
        if not active.any():
            return 0.0
        return float(np.max(stds[active] / means[active]))

    @staticmethod
    def _coefficient_of_variation(arr: np.ndarray) -> float:
        mean = arr.mean()
        if mean == 0:
            return 0.0
        return float(arr.std() / mean)

    def _write_progress_row(self, metrics, is_stable):
        """
        # OUTPUT SPECIFICATION
        path: /equilibrium_progress.csv
        filetype: csv
        category: log
        description: Written only if EQUILIBRIUM_TERMINATION is True. One row per equilibrium check (every EQUILIBRIUM_CHECK_RATE steps), regardless of whether equilibrium has been reached yet -- lets you inspect how close the population is to equilibrium at any point, including separate stability metrics for phenotypes at ages before reproductive maturity and from reproductive maturity up to the midpoint of the reproductive age range.
        trait granularity: population summary (popsize and phenotype coefficients of variation, overall and by age band, across the check window)
        time granularity: snapshot
        frequency parameter: EQUILIBRIUM_CHECK_RATE
        structure: A table with columns [step, popsize, popsize_cv, phenotype_cv_overall, phenotype_cv_immature, phenotype_cv_early_mature, n_checks_so_far, equilibrium_reached].
        header: step,popsize,popsize_cv,phenotype_cv_overall,phenotype_cv_immature,phenotype_cv_early_mature,n_checks_so_far,equilibrium_reached
        """
        if metrics is None:
            row_metrics = [float("nan")] * 4
        else:
            row_metrics = [
                metrics["popsize_cv"],
                metrics["phenotype_cv_overall"],
                metrics["phenotype_cv_immature"],
                metrics["phenotype_cv_early_mature"],
            ]
        with open(self.odir / "equilibrium_progress.csv", "a", newline="") as f:
            csv.writer(f).writerow(
                [variables.steps, self.popsize_history[-1], *row_metrics, len(self.popsize_history), is_stable]
            )

    def _write_summary(self, metrics: dict, new_end: int, steps_saved: int):
        """
        # OUTPUT SPECIFICATION
        path: /equilibrium_summary.json
        filetype: json
        category: log
        description: Written only if EQUILIBRIUM_TERMINATION is True and equilibrium was reached. Documents the step at which the population was judged to have reached equilibrium, the stability metrics (population size and age-banded phenotype coefficients of variation) at that point, and the shortened STEPS_PER_SIMULATION (original end, new end, and steps saved) used to wind the run down cleanly.
        trait granularity: N/A
        time granularity: N/A
        frequency parameter: once
        structure: A json dictionary.
        header: None
        """
        summary = {
            "equilibrium_reached": True,
            "step_reached": variables.steps,
            "original_steps_per_simulation": self.original_steps_per_simulation,
            "new_steps_per_simulation": new_end,
            "steps_saved": steps_saved,
            "check_rate": parametermanager.parameters.EQUILIBRIUM_CHECK_RATE,
            "window": self.window,
            "tolerance": self.tolerance,
            "maturation_age": self.maturation_age,
            "mid_age": self.mid_age,
            **metrics,
            "popsize_history": [int(x) for x in self.popsize_history],
        }
        with open(self.odir / "equilibrium_summary.json", "w") as f:
            json.dump(summary, f, indent=4)
