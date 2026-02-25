"""Functional tests for genome and phenotype array structure.

Verifies that genome shape (N, ploidy, n_loci, bits_per_locus) and
phenotype shape (N, expected_phenotype_length) are correct under
different AGE_LIMIT, evolvable trait, and agespecific configurations.
"""

import numpy as np
import pandas as pd
import pytest
import yaml

import aegis_sim
from aegis_sim import parameterization, variables
from aegis_sim.dataclasses.population import Population


def _run_sim(tmp_path, config_overrides, name="test"):
    """Run a short simulation and return (odir, final parameterization state)."""
    base = {
        "STEPS_PER_SIMULATION": 5,
        "LOGGING_RATE": 100,
        "INTERVAL_RATE": 100,
        "SNAPSHOT_RATE": 1,
        "PICKLE_RATE": 100,
        "POPGENSTATS_RATE": 100,
        "TE_RATE": 100,
        "TE_DURATION": 50,
        "INITIAL_POPULATION_SIZE": 500,
        "CARRYING_CAPACITY_EGGS": 1000,
        "RESOURCE_ADDITIVE_GROWTH": 1000,
        "MATURATION_AGE": 1,
        "SNAPSHOT_FINAL_COUNT": 1,
    }
    base.update(config_overrides)
    config_path = tmp_path / f"{name}.yml"
    with open(config_path, "w") as f:
        yaml.dump(base, f)
    aegis_sim.run(
        custom_config_path=config_path,
        pickle_path=None,
        overwrite=False,
        custom_input_params={},
    )
    return tmp_path / name


# ---------------------------------------------------------------------------
# AGE_LIMIT variations
# ---------------------------------------------------------------------------

class TestAgeLimitAffectsShape:
    """Genome n_loci and phenotype length scale with AGE_LIMIT when traits are age-specific."""

    @pytest.mark.parametrize("age_limit", [10, 30, 50])
    def test_phenotype_columns_scale_with_age_limit(self, tmp_path, age_limit):
        """With defaults (surv+repr evolvable+agespecific), phenotype width = 2 * AGE_LIMIT."""
        odir = _run_sim(tmp_path, {"AGE_LIMIT": age_limit}, name=f"age{age_limit}")
        pheno_dir = odir / "snapshots" / "phenotypes"
        feather_files = sorted(pheno_dir.glob("*.feather"))
        assert len(feather_files) > 0, "No phenotype snapshot feather files found"
        df = pd.read_feather(feather_files[0])
        # Default: surv evolvable+agespecific, repr evolvable+agespecific
        expected_pheno_len = 2 * age_limit
        assert df.shape[1] == expected_pheno_len, (
            f"Expected {expected_pheno_len} phenotype columns for AGE_LIMIT={age_limit}, "
            f"got {df.shape[1]}"
        )

    @pytest.mark.parametrize("age_limit", [10, 30, 50])
    def test_genotype_columns_scale_with_age_limit(self, tmp_path, age_limit):
        """With defaults, genotype width = 2 * AGE_LIMIT * BITS_PER_LOCUS (surv+repr loci)."""
        odir = _run_sim(tmp_path, {"AGE_LIMIT": age_limit, "BITS_PER_LOCUS": 1}, name=f"geno_age{age_limit}")
        geno_dir = odir / "snapshots" / "genotypes"
        feather_files = sorted(geno_dir.glob("*.feather"))
        assert len(feather_files) > 0
        df = pd.read_feather(feather_files[0])
        # 2 evolvable agespecific traits * AGE_LIMIT loci * 1 bit * ploidy(2)
        expected_geno_len = 2 * age_limit * 1 * 2
        assert df.shape[1] == expected_geno_len, (
            f"Expected {expected_geno_len} genotype columns for AGE_LIMIT={age_limit}, "
            f"got {df.shape[1]}"
        )


# ---------------------------------------------------------------------------
# Evolvable trait variations
# ---------------------------------------------------------------------------

class TestEvolvableTraits:
    """Phenotype length changes when traits are toggled evolvable/non-evolvable."""

    def test_all_five_traits_evolvable_agespecific(self, tmp_path):
        """All 5 traits evolvable + agespecific: phenotype width = 5 * AGE_LIMIT."""
        age_limit = 20
        odir = _run_sim(tmp_path, {
            "AGE_LIMIT": age_limit,
            "G_surv_evolvable": True, "G_surv_agespecific": True,
            "G_repr_evolvable": True, "G_repr_agespecific": True,
            "G_neut_evolvable": True, "G_neut_agespecific": True,
            "G_muta_evolvable": True, "G_muta_agespecific": True,
            "G_grow_evolvable": True, "G_grow_agespecific": True,
        }, name="all5_agespec")
        pheno_dir = odir / "snapshots" / "phenotypes"
        feather_files = sorted(pheno_dir.glob("*.feather"))
        assert len(feather_files) > 0
        df = pd.read_feather(feather_files[0])
        expected = 5 * age_limit
        assert df.shape[1] == expected, f"Expected {expected}, got {df.shape[1]}"

    def test_only_surv_evolvable(self, tmp_path):
        """Only surv evolvable + agespecific: phenotype width = AGE_LIMIT."""
        age_limit = 20
        odir = _run_sim(tmp_path, {
            "AGE_LIMIT": age_limit,
            "G_surv_evolvable": True, "G_surv_agespecific": True,
            "G_repr_evolvable": False,
            "G_neut_evolvable": False,
            "G_muta_evolvable": False,
            "G_grow_evolvable": False,
        }, name="surv_only")
        pheno_dir = odir / "snapshots" / "phenotypes"
        feather_files = sorted(pheno_dir.glob("*.feather"))
        assert len(feather_files) > 0
        df = pd.read_feather(feather_files[0])
        assert df.shape[1] == age_limit

    @pytest.mark.xfail(reason="Bug: PopgenStats.get_genomes_sample crashes on empty genomes when no traits are evolvable", strict=True)
    def test_no_traits_evolvable(self, tmp_path):
        """No evolvable traits: phenotype width = 0."""
        odir = _run_sim(tmp_path, {
            "AGE_LIMIT": 20,
            "G_surv_evolvable": False,
            "G_repr_evolvable": False,
            "G_neut_evolvable": False,
            "G_muta_evolvable": False,
            "G_grow_evolvable": False,
        }, name="none_evolvable")
        pheno_dir = odir / "snapshots" / "phenotypes"
        feather_files = sorted(pheno_dir.glob("*.feather"))
        assert len(feather_files) > 0
        df = pd.read_feather(feather_files[0])
        assert df.shape[1] == 0


# ---------------------------------------------------------------------------
# Age-specificity variations
# ---------------------------------------------------------------------------

class TestAgeSpecificity:
    """Phenotype length depends on whether traits are age-specific or not."""

    def test_surv_not_agespecific(self, tmp_path):
        """surv evolvable but NOT agespecific: contributes 1 column instead of AGE_LIMIT."""
        age_limit = 30
        odir = _run_sim(tmp_path, {
            "AGE_LIMIT": age_limit,
            "G_surv_evolvable": True, "G_surv_agespecific": False,
            "G_repr_evolvable": True, "G_repr_agespecific": True,
            "G_neut_evolvable": False,
            "G_muta_evolvable": False,
            "G_grow_evolvable": False,
        }, name="surv_noage")
        pheno_dir = odir / "snapshots" / "phenotypes"
        feather_files = sorted(pheno_dir.glob("*.feather"))
        assert len(feather_files) > 0
        df = pd.read_feather(feather_files[0])
        # surv: 1 col (not agespecific) + repr: AGE_LIMIT cols
        expected = 1 + age_limit
        assert df.shape[1] == expected, f"Expected {expected}, got {df.shape[1]}"

    def test_both_not_agespecific(self, tmp_path):
        """surv + repr evolvable but NOT agespecific: 2 phenotype columns total."""
        odir = _run_sim(tmp_path, {
            "AGE_LIMIT": 40,
            "G_surv_evolvable": True, "G_surv_agespecific": False,
            "G_repr_evolvable": True, "G_repr_agespecific": False,
            "G_neut_evolvable": False,
            "G_muta_evolvable": False,
            "G_grow_evolvable": False,
        }, name="both_noage")
        pheno_dir = odir / "snapshots" / "phenotypes"
        feather_files = sorted(pheno_dir.glob("*.feather"))
        assert len(feather_files) > 0
        df = pd.read_feather(feather_files[0])
        assert df.shape[1] == 2


# ---------------------------------------------------------------------------
# Phenotype value ranges
# ---------------------------------------------------------------------------

class TestPhenotypeValueRanges:
    """Phenotype values should be within [lo, hi] bounds for each trait."""

    def test_phenotype_values_in_unit_interval(self, tmp_path):
        """Default lo=0, hi=1 for surv; lo=0, hi=0.5 for repr."""
        odir = _run_sim(tmp_path, {"AGE_LIMIT": 20}, name="pheno_range")
        pheno_dir = odir / "snapshots" / "phenotypes"
        feather_files = sorted(pheno_dir.glob("*.feather"))
        assert len(feather_files) > 0
        df = pd.read_feather(feather_files[0])
        if df.shape[1] > 0:
            assert np.all(df.values >= 0), "Phenotype values should be >= 0"
            assert np.all(df.values <= 1), "Phenotype values should be <= 1"


# ---------------------------------------------------------------------------
# BITS_PER_LOCUS variations
# ---------------------------------------------------------------------------

class TestBitsPerLocus:
    """Genome width scales with BITS_PER_LOCUS."""

    @pytest.mark.parametrize("bpl", [1, 4, 8])
    def test_genotype_width_scales_with_bpl(self, tmp_path, bpl):
        """Genotype columns = n_loci * BITS_PER_LOCUS * ploidy."""
        age_limit = 10
        odir = _run_sim(tmp_path, {
            "AGE_LIMIT": age_limit,
            "BITS_PER_LOCUS": bpl,
            "G_surv_evolvable": True, "G_surv_agespecific": True,
            "G_repr_evolvable": False,
            "G_neut_evolvable": False,
            "G_muta_evolvable": False,
            "G_grow_evolvable": False,
        }, name=f"bpl{bpl}")
        geno_dir = odir / "snapshots" / "genotypes"
        feather_files = sorted(geno_dir.glob("*.feather"))
        assert len(feather_files) > 0
        df = pd.read_feather(feather_files[0])
        # 1 evolvable agespecific trait * AGE_LIMIT loci * bpl bits * ploidy(2)
        expected = age_limit * bpl * 2
        assert df.shape[1] == expected, f"Expected {expected}, got {df.shape[1]}"
