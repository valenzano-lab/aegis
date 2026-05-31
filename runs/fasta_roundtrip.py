"""Round-trip validation for FASTA export.

Loads the final-step pickle from runs/fasta_test, encodes the population's
genomes to FASTA + mapping, decodes back, and asserts:
  (1) decoded genome bits are exactly equal to the originals,
  (2) phenotypes derived from the decoded genomes via `architect`
      are numerically identical to the phenotypes stored on the pickle.

Also produces a side-by-side mean-surv/repr plot at runs/fasta_test/roundtrip.png.
"""

import pathlib
import sys

import numpy as np

from aegis_sim import parameterization, submodels, variables
from aegis_sim.dataclasses.genomes import Genomes
from aegis_sim.dataclasses.population import Population
from aegis_sim.parameterization import parametermanager
from aegis_sim.utilities.fasta import (
    decode_fasta_to_genomes,
    encode_population_to_fasta,
)


def init_without_recording(config_path: pathlib.Path) -> None:
    """Minimal init: set up params/variables/submodels (incl. architect) but
    do not create or touch the recording output directory."""
    parametermanager.init(custom_config_path=config_path, custom_input_params={})
    variables.init(
        variables,
        custom_config_path=config_path,
        pickle_path=None,
        RANDOM_SEED=parametermanager.parameters.RANDOM_SEED,
    )
    parameterization.init_traits(parameterization)
    submodels.init(submodels, parametermanager=parametermanager)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "runs" / "fasta_test.yml"
OUTPUT_DIR = REPO_ROOT / "runs" / "fasta_test"
PICKLE_DIR = OUTPUT_DIR / "pickles"
FASTA_DIR = OUTPUT_DIR / "fasta"


def latest_pickle(pickle_dir: pathlib.Path) -> pathlib.Path:
    files = [p for p in pickle_dir.iterdir() if p.name.isdigit()]
    if not files:
        raise FileNotFoundError(f"No numeric pickles in {pickle_dir}")
    return max(files, key=lambda p: int(p.name))


def main() -> int:
    init_without_recording(CONFIG_PATH)

    pickle_path = latest_pickle(PICKLE_DIR)
    print(f"Loading pickle: {pickle_path}")
    population = Population.load_pickle_from(pickle_path)
    print(f"  n={len(population)}  genome shape={population.genomes.array.shape}")

    fasta_path, mapping_path = encode_population_to_fasta(
        population, FASTA_DIR, name="step{}".format(pickle_path.name), mask_seed=2026
    )
    fasta_kb = fasta_path.stat().st_size / 1024
    print(f"Wrote {fasta_path.name} ({fasta_kb:.1f} KB) and {mapping_path.name}")

    decoded, record_ids = decode_fasta_to_genomes(fasta_path, mapping_path)
    print(f"Decoded {len(record_ids)} records, shape={decoded.shape}")

    # (1) bit equality
    original = population.genomes.array
    if decoded.shape != original.shape:
        print(f"FAIL shape: {decoded.shape} vs {original.shape}")
        return 1
    diffs = int((decoded != original).sum())
    if diffs:
        print(f"FAIL bits: {diffs} differing bits out of {decoded.size}")
        return 1
    print(f"OK bits: all {decoded.size} bits identical across {len(decoded)} individuals")

    # (2) phenotype equality
    decoded_phenotypes = submodels.architect(Genomes(decoded))
    original_phenotypes = population.phenotypes
    max_abs_diff = float(np.abs(decoded_phenotypes.array - original_phenotypes.array).max())
    if not np.allclose(decoded_phenotypes.array, original_phenotypes.array, atol=1e-12):
        print(f"FAIL phenotypes: max |diff| = {max_abs_diff:.2e}")
        return 1
    print(f"OK phenotypes: max |diff| = {max_abs_diff:.2e}")

    # Side-by-side plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from aegis_sim.parameterization import parametermanager

        age_limit = parametermanager.parameters.AGE_LIMIT
        # composite phenotype layout: 5 traits × AGE_LIMIT each, in trait order surv/repr/neut/muta/grow.
        ages = np.arange(age_limit)

        def mean_curve(phen_array, trait_idx):
            start = trait_idx * age_limit
            end = start + age_limit
            return phen_array[:, start:end].mean(axis=0)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for ax, trait_idx, title in [(axes[0], 0, "surv"), (axes[1], 1, "repr")]:
            ax.plot(ages, mean_curve(original_phenotypes.array, trait_idx), label="pickle", lw=2)
            ax.plot(
                ages,
                mean_curve(decoded_phenotypes.array, trait_idx),
                label="FASTA → decode → architect",
                lw=1,
                ls="--",
            )
            ax.set_title(f"mean {title}(age)")
            ax.set_xlabel("age")
            ax.legend()
        axes[0].set_ylabel("phenotype")
        fig.suptitle(f"Round-trip validation, n={len(population)}")
        fig.tight_layout()
        plot_path = OUTPUT_DIR / "roundtrip.png"
        fig.savefig(plot_path, dpi=120)
        print(f"Wrote plot: {plot_path}")
    except Exception as exc:
        print(f"(plot skipped: {exc})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
