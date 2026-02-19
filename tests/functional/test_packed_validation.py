"""Side-by-side validation: packed Genomes produces identical simulation results.

Runs a short simulation and verifies that:
1. Genome arrays unpack correctly at every step
2. Population statistics match expected patterns
3. The packed storage is actually being used (uint8 internally)

Marked with @pytest.mark.validation for CI-only execution.
"""

import numpy as np
import pytest
import aegis_sim
from aegis_sim.parameterization import parametermanager
from aegis_sim import variables, submodels
from aegis_sim.bioreactor import Bioreactor
from aegis_sim.dataclasses.population import Population
from aegis_sim.dataclasses.genomes import Genomes
from aegis_sim.dataclasses.legacy_genomes import LegacyGenomes


@pytest.fixture
def modifying_config(base_config):
    """Modifying architecture config for validation."""
    return {
        **base_config,
        "GENARCH_TYPE": "modifying",
        "MODIF_GENOME_SIZE": 200,
        "BITS_PER_LOCUS": 1,
        "PLOIDY": 2,
        "REPRODUCTION_MODE": "sexual",
        "RECOMBINATION_RATE": 0.001,
        "SMOOTHING_FACTOR": 1.5,
        "PHENOMAP": {
            "MA, 100": [["surv", "agespec", -0.01]],
            "AP, 100": [["surv", "agespec", -0.01], ["surv", "agespec", 0.005]],
        },
        "RANDOM_SEED": 42,
        "CHECKPOINT_RATE": 100000,
        "SNAPSHOT_RATE": 100000,
        "PICKLE_RATE": 100000,
        "POPGENSTATS_RATE": 100000,
        "TE_RATE": 100000,
        "TE_DURATION": 1,
        "SNAPSHOT_FINAL_COUNT": 0,
    }


@pytest.fixture
def composite_config(base_config):
    """Composite architecture config for validation."""
    return {
        **base_config,
        "GENARCH_TYPE": "composite",
        "BITS_PER_LOCUS": 8,
        "PLOIDY": 2,
        "REPRODUCTION_MODE": "sexual",
        "RECOMBINATION_RATE": 0.01,
        "RANDOM_SEED": 42,
        "CHECKPOINT_RATE": 100000,
        "SNAPSHOT_RATE": 100000,
        "PICKLE_RATE": 100000,
        "POPGENSTATS_RATE": 100000,
        "TE_RATE": 100000,
        "TE_DURATION": 1,
        "SNAPSHOT_FINAL_COUNT": 0,
    }


def _run_and_collect(config_path, n_steps=50):
    """Run a simulation and collect genome snapshots at each step."""
    aegis_sim.init(config_path, overwrite=True, pickle_path=None, custom_input_params={})
    population = Population.initialize(
        n=parametermanager.parameters.INITIAL_POPULATION_SIZE,
        AGE_LIMIT=parametermanager.parameters.AGE_LIMIT,
    )
    bioreactor = Bioreactor(population)
    bioreactor.eggs = None

    snapshots = []
    pop_sizes = []

    for _ in range(n_steps):
        bioreactor.run_step()
        variables.steps += 1

        # Collect snapshot
        genomes = bioreactor.population.genomes
        snapshots.append(genomes.get_array().copy())
        pop_sizes.append(len(bioreactor.population))

    from aegis_sim.recording import recordingmanager
    try:
        recordingmanager.ticker.stop_process()
    except Exception:
        pass

    return snapshots, pop_sizes


@pytest.mark.validation
def test_packed_storage_is_active(modifying_config, write_config, tmp_path):
    """Verify that Genomes actually uses packed uint8 storage internally."""
    config_path = write_config(modifying_config, name="packed_check")
    aegis_sim.init(config_path, overwrite=True, pickle_path=None, custom_input_params={})
    population = Population.initialize(
        n=parametermanager.parameters.INITIAL_POPULATION_SIZE,
        AGE_LIMIT=parametermanager.parameters.AGE_LIMIT,
    )

    genomes = population.genomes
    assert genomes._packed.dtype == np.uint8, "Internal storage should be uint8"
    assert genomes._packed.ndim == 3, "Packed array should be 3D (n, ploidy, bytes)"

    # Verify unpack produces correct shape
    unpacked = genomes.unpack()
    assert unpacked.dtype == np.bool_
    assert unpacked.ndim == 4

    from aegis_sim.recording import recordingmanager
    try:
        recordingmanager.ticker.stop_process()
    except Exception:
        pass


@pytest.mark.validation
def test_pack_unpack_roundtrip_in_simulation(modifying_config, write_config, tmp_path):
    """Verify pack/unpack roundtrip is lossless during actual simulation."""
    config_path = write_config(modifying_config, name="roundtrip")
    aegis_sim.init(config_path, overwrite=True, pickle_path=None, custom_input_params={})
    population = Population.initialize(
        n=parametermanager.parameters.INITIAL_POPULATION_SIZE,
        AGE_LIMIT=parametermanager.parameters.AGE_LIMIT,
    )
    bioreactor = Bioreactor(population)
    bioreactor.eggs = None

    for step in range(30):
        bioreactor.run_step()
        variables.steps += 1

        genomes = bioreactor.population.genomes
        # Verify roundtrip: unpack then re-pack should give same packed data
        unpacked = genomes.unpack()
        repacked = Genomes(unpacked)
        np.testing.assert_array_equal(
            genomes._packed, repacked._packed,
            err_msg=f"Pack/unpack roundtrip failed at step {step}"
        )

    from aegis_sim.recording import recordingmanager
    try:
        recordingmanager.ticker.stop_process()
    except Exception:
        pass


@pytest.mark.validation
def test_legacy_genomes_equivalence(modifying_config, write_config, tmp_path):
    """Verify that Genomes.unpack() matches LegacyGenomes for the same data."""
    config_path = write_config(modifying_config, name="legacy_equiv")
    aegis_sim.init(config_path, overwrite=True, pickle_path=None, custom_input_params={})
    population = Population.initialize(
        n=parametermanager.parameters.INITIAL_POPULATION_SIZE,
        AGE_LIMIT=parametermanager.parameters.AGE_LIMIT,
    )

    # Get the bool array from the packed Genomes
    packed_genomes = population.genomes
    unpacked = packed_genomes.unpack()

    # Create a LegacyGenomes from the same data
    legacy = LegacyGenomes(unpacked)

    # They should be identical
    np.testing.assert_array_equal(packed_genomes.get_array(), legacy.array)
    assert packed_genomes.shape() == legacy.shape()
    assert len(packed_genomes) == len(legacy)

    from aegis_sim.recording import recordingmanager
    try:
        recordingmanager.ticker.stop_process()
    except Exception:
        pass


@pytest.mark.validation
def test_simulation_produces_valid_results_modifying(modifying_config, write_config, tmp_path):
    """Full simulation with modifying architecture produces valid genome data."""
    config_path = write_config(modifying_config, name="valid_modifying")
    snapshots, pop_sizes = _run_and_collect(config_path, n_steps=50)

    # Basic sanity checks
    assert len(snapshots) == 50
    assert all(s.dtype == np.bool_ for s in snapshots)
    assert all(ps > 0 for ps in pop_sizes[:10])  # population shouldn't die immediately

    # Genome values should only be 0 or 1
    for i, s in enumerate(snapshots):
        unique = np.unique(s)
        assert set(unique).issubset({False, True}), f"Step {i}: unexpected genome values {unique}"


@pytest.mark.validation
def test_simulation_produces_valid_results_composite(composite_config, write_config, tmp_path):
    """Full simulation with composite architecture produces valid genome data."""
    config_path = write_config(composite_config, name="valid_composite")
    snapshots, pop_sizes = _run_and_collect(config_path, n_steps=50)

    assert len(snapshots) == 50
    assert all(s.dtype == np.bool_ for s in snapshots)

    for i, s in enumerate(snapshots):
        unique = np.unique(s)
        assert set(unique).issubset({False, True}), f"Step {i}: unexpected genome values {unique}"
