"""Functional test: phenodiff and phenodiff_accelerated produce identical results.

Runs a short simulation with the modifying architecture (PHENOMAP config),
then calls both phenodiff methods on the same population and verifies they
produce the same phenotype array. This tests the full pipeline including
trait index resolution from parameterization.
"""

import numpy as np
import pytest
import aegis_sim
from aegis_sim.parameterization import parametermanager
from aegis_sim.dataclasses.population import Population
from aegis_sim import submodels


@pytest.fixture
def modifying_config(base_config):
    """Config using the modifying architecture with a PHENOMAP."""
    return {
        **base_config,
        "GENARCH_TYPE": "modifying",
        "MODIF_GENOME_SIZE": 200,
        "BITS_PER_LOCUS": 1,
        "PLOIDY": 2,
        "REPRODUCTION_MODE": "sexual",
        "RECOMBINATION_RATE": 0.001,
        "SMOOTHING_FACTOR": 0,  # disable smoothing to isolate phenodiff
        "PHENOMAP": {
            "MA, 100": [
                ["surv", "agespec", -0.01],
            ],
            "AP, 100": [
                ["surv", "agespec", -0.01],
                ["surv", "agespec", 0.005],
            ],
        },
        "RANDOM_SEED": 42,
    }


def test_phenodiff_matches_accelerated(modifying_config, write_config, tmp_path):
    """phenodiff and phenodiff_accelerated must produce identical phenotype arrays."""
    config_path = write_config(modifying_config, name="gpm_test")

    aegis_sim.init(
        config_path,
        overwrite=True,
        pickle_path=None,
        custom_input_params={},
    )

    population = Population.initialize(
        n=parametermanager.parameters.INITIAL_POPULATION_SIZE,
        AGE_LIMIT=parametermanager.parameters.AGE_LIMIT,
    )

    # Get the GPM and architecture
    architect = submodels.architect
    gpm = architect.architecture.phenomap

    # Prepare inputs (same as what architect.__call__ does)
    from aegis_sim.submodels.genetics import ploider
    genomes = population.genomes.get_array()
    if genomes.shape[1] == 1:
        haploid = genomes[:, 0]
    else:
        haploid = ploider.ploider.diploid_to_haploid(genomes)

    interpretome = haploid.reshape(len(population), -1)

    from aegis_sim.dataclasses.phenotypes import Phenotypes
    zeropheno = Phenotypes.init_phenotype_array(popsize=len(population)).array

    # Call both methods
    result_slow = gpm.phenodiff(vectors=interpretome, zeropheno=zeropheno.copy())
    result_fast = gpm.phenodiff_accelerated(vectors=interpretome, zeropheno=zeropheno.copy())

    np.testing.assert_allclose(result_slow, result_fast, atol=1e-9)


def test_phenodiff_nonzero_output(modifying_config, write_config, tmp_path):
    """phenodiff_accelerated should produce nonzero phenotype changes
    (sanity check that the phenomap is actually doing something)."""
    config_path = write_config(modifying_config, name="gpm_nonzero")

    aegis_sim.init(
        config_path,
        overwrite=True,
        pickle_path=None,
        custom_input_params={},
    )

    population = Population.initialize(
        n=parametermanager.parameters.INITIAL_POPULATION_SIZE,
        AGE_LIMIT=parametermanager.parameters.AGE_LIMIT,
    )

    architect = submodels.architect
    gpm = architect.architecture.phenomap

    from aegis_sim.submodels.genetics import ploider
    genomes = population.genomes.get_array()
    haploid = ploider.ploider.diploid_to_haploid(genomes)
    interpretome = haploid.reshape(len(population), -1)

    from aegis_sim.dataclasses.phenotypes import Phenotypes
    zeropheno = Phenotypes.init_phenotype_array(popsize=len(population)).array

    result = gpm.phenodiff_accelerated(vectors=interpretome, zeropheno=zeropheno.copy())

    # The phenomap should have produced some nonzero changes
    assert np.any(result != 0), "phenodiff_accelerated produced all zeros — phenomap may not be wired correctly"
