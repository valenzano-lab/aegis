import pytest
from aegis.modules.population import Population


@pytest.mark.parametrize(
    "list_,len_",
    [
        ([1] * 1, 1),
        ([1] * 10, 10),
        ([1] * 42, 42),
        ([1] * 100, 100),
    ],
)
def test_len(list_, len_):
    population = Population(list_, list_, list_, list_, list_)
    assert len(population) == len_


@pytest.mark.parametrize(
    "genomes,ages,births,birthdays,phenotypes",
    [
        ([1] * 5, [1] * 4, [1] * 4, [1] * 4, [1] * 4),
        ([1] * 3, [1] * 4, [1] * 4, [1] * 4, [1] * 4),
        ([1] * 4, [1] * 40, [1] * 4, [1] * 4, [1] * 4),
        ([1] * 4, [1] * 4, [1] * 50, [1] * 4, [1] * 4),
        ([1] * 4, [1] * 4, [1] * 4, [1] * 40, [1] * 4),
        ([1] * 4, [1] * 4, [1] * 4, [1] * 4, [1] * 40),
    ],
)
def test_init(genomes, ages, births, birthdays, phenotypes):
    with pytest.raises(ValueError):
        _ = Population(genomes, ages, births, birthdays, phenotypes)
