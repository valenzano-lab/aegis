"""
Ensure that these tests run after other tests have run, so that there are simulation data to test against. This is why the module is called test_Zcontainer.
"""

import pytest
import pathlib
import logging
import pandas as pd

from aegis_sim.utilities.container import Container
from test_sim import test_experiment_path


def get_experiment_subfolders():
    return [folder for folder in test_experiment_path.iterdir() if folder.is_dir()]


@pytest.fixture(params=get_experiment_subfolders())
def container_instance(request):
    return Container(request.param)


def test_get_path(container_instance: Container):
    _ = container_instance.get_path(name="phenotypes")
    _ = container_instance.get_path(name="genotypes")
    _ = container_instance.get_path(name="log")
    _ = container_instance.get_path(name="ticker")
    _ = container_instance.get_path(name="output_summary")
    _ = container_instance.get_path(name="input_summary")
    _ = container_instance.get_path(name="envdriftmap")
    _ = container_instance.get_path(name="snapshots")
    _ = container_instance.get_path(name="pickles")
    # _ = container_instance.get_path(name="te")
    _ = container_instance.get_path(name="popsize_before_reproduction")
    _ = container_instance.get_path(name="popsize_after_reproduction")
    _ = container_instance.get_path(name="eggnum_after_reproduction")
    _ = container_instance.get_path(name="phenomap")


def test_get_record_structure(container_instance: Container):
    _ = container_instance.get_record_structure()


def test_get_phenomap(container_instance: Container):
    _ = container_instance.get_phenomap()


def test_get_log(container_instance: Container):
    _ = container_instance.get_log()
    assert isinstance(_, pd.DataFrame)


def test_get_simple_log(container_instance: Container):
    _ = container_instance.get_simple_log()


def test_get_ticker(container_instance: Container):
    _ = container_instance.get_ticker()


def test_get_config(container_instance: Container):
    _ = container_instance.get_config()


def test_get_final_config(container_instance: Container):
    _ = container_instance.get_final_config()


def test_get_generations_until_interval(container_instance: Container):
    _ = container_instance.get_generations_until_interval()


def test_get_output_summary(container_instance: Container):
    _ = container_instance.get_output_summary()


def test_get_input_summary(container_instance: Container):
    _ = container_instance.get_input_summary()


def test_get_envidriftmap(container_instance: Container):
    _ = container_instance.get_envidriftmap()


def test_get_birth_table_observed_interval(container_instance: Container):
    _ = container_instance.get_birth_table_observed_interval()


def test_get_life_table_observed_interval(container_instance: Container):
    _ = container_instance.get_life_table_observed_interval()


def test_get_life_table_observed_snapshot(container_instance: Container):
    _ = container_instance.get_life_table_observed_snapshot(record_index=0)
    _ = container_instance.get_life_table_observed_snapshot(record_index=-1)


def test_get_death_table_observed_interval(container_instance: Container):
    _ = container_instance.get_death_table_observed_interval()


def test_get_surv_observed_interval(container_instance: Container):
    _ = container_instance.get_surv_observed_interval()


def test_get_fert_observed_interval(container_instance: Container):
    _ = container_instance.get_fert_observed_interval()


def test_get_genotypes_intrinsic_snapshot(container_instance: Container):
    _ = container_instance.get_genotypes_intrinsic_snapshot(record_index=0)
    _ = container_instance.get_genotypes_intrinsic_snapshot(record_index=-1)


def test_get_phenotype_intrinsic_snapshot(container_instance: Container):
    _ = container_instance.get_phenotype_intrinsic_snapshot(record_index=0, trait=None)
    _ = container_instance.get_phenotype_intrinsic_snapshot(record_index=-1, trait=None)
    # TODO make trait specific?


def test_get_demography_observed_snapshot(container_instance: Container):
    _ = container_instance.get_demography_observed_snapshot(record_index=0)
    _ = container_instance.get_demography_observed_snapshot(record_index=-1)


def test_get_genotypes_intrinsic_interval(container_instance: Container):
    _ = container_instance.get_genotypes_intrinsic_interval()


def test_get_phenotype_intrinsic_interval(container_instance: Container):
    _ = container_instance.get_phenotype_intrinsic_interval(trait="surv")
    _ = container_instance.get_phenotype_intrinsic_interval(trait="repr")
    _ = container_instance.get_phenotype_intrinsic_interval(trait="muta")
    _ = container_instance.get_phenotype_intrinsic_interval(trait="grow")


# def test_get_survival_analysis_TE_observed_interval(container_instance: Container):
#     _ = container_instance.get_survival_analysis_TE_observed_interval(record_index=0)
#     _ = container_instance.get_survival_analysis_TE_observed_interval(record_index=-1)


def test_get_population_size_before_reproduction(container_instance: Container):
    _ = container_instance.get_population_size_before_reproduction()


def test_get_population_size_after_reproduction(container_instance: Container):
    _ = container_instance.get_population_size_after_reproduction()


def test_get_egg_number_after_reproduction(container_instance: Container):
    _ = container_instance.get_egg_number_after_reproduction()


def test_get_resource_amount_before_scavenging(container_instance: Container):
    _ = container_instance.get_resource_amount_before_scavenging()


def test_get_resource_amount_after_scavenging(container_instance: Container):
    _ = container_instance.get_resource_amount_after_scavenging()


def test_get_lifetime_reproduction(container_instance: Container):
    _ = container_instance.get_lifetime_reproduction()


def test_get_average_age_at_reproduction(container_instance: Container):
    _ = container_instance.get_average_age_at_reproduction()
