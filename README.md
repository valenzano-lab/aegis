<style>
details{
    margin-bottom:12px;
}
summary {
    margin-bottom:10px;
    font-style:italic;
    cursor:pointer;
}
</style>

[![PyPI version](https://badge.fury.io/py/aegis-sim.svg)](https://badge.fury.io/py/aegis-sim)
[![Python 3.6+](https://img.shields.io/badge/python-3.6%2B-blue)](https://www.python.org/downloads/release/python-360/)

# AEGIS

> Aging of Evolving Genomes In Silico (EY-jis, /eɪd͡ʒɪs/)

Numerical model for life history evolution of age-structured populations under customizable ecological scenarios.

## How to install

We recommend that you install `aegis-sim` from [PyPI](https://pypi.org/project/aegis-sim/) into a [virtual environment](https://docs.python.org/3/library/venv.html).

```bash
$ pip install aegis-sim
```

<details>
  <summary>Cheat sheet</summary>

```bash
# Unix/macOS
python3 -m venv aegis-venv
. aegis-venv/bin/activate
python3 -m pip install aegis-sim
```
```bash
# Windows
python -m venv aegis-venv
.\aegis-venv\Scripts\activate
python -m pip install aegis-sim
```
</details>

<details> 
<summary>For developers</summary>

```bash
# Unix/macOS
. {path}/bin/activate # Activate virtual environment
git clone git@github.com:valenzano-lab/aegis.git # Clone repo
cd aegis
python3 -m build # Local build
python3 -m pip install -e . # Local editable install
```
</details>

## How to run

1. __Create a configuration file.__

    Before running a custom AEGIS simulation, you must create a configuration file which will contain your
    custom parameter values. The file must be in [YAML](https://en.wikipedia.org/wiki/YAML) format.
    Default parameters are set in file [default.yml](src/aegis/parameters/default.yml).
    List of modifiable parameters, and all relevant details can be found in the [wiki](https://github.com/valenzano-lab/aegis/wiki/Input).

    An example of a custom YML file:
    ```yml
    # custom.yml

    RANDOM_SEED_: 42
    STAGES_PER_SIMULATION_: 10000
    MAX_LIFESPAN: 50
    ```


1. __Start the simulation.__

    ```sh
    $ aegis {path/to/file}.yml # In this case, `aegis custom.yml`
    ```


1. __Inspect the output.__

    Output files will be created in the `{path/to/file}` directory (in this case, in the `custom` directory). 
    Detailed description of the content and format of output files can be found in the [wiki](https://github.com/valenzano-lab/aegis/wiki/Output).
    

## Related articles

- [An In Silico Model to Simulate the Evolution of Biological Aging (2016)](https://www.biorxiv.org/content/10.1101/037952v1)
- [AEGIS: An In Silico Tool to model Genome Evolution in Age-Structured Populations (2019)](https://www.biorxiv.org/content/10.1101/646877v1)

## Authors

- **Martin Bagić**: [email](martin.bagic@outlook.com), [github](https://github.com/martinbagic)
- **Arian Šajina**: [github](https://github.com/ariansajina)
- **William Bradshaw**: [github](https://github.com/willbradshaw)
- **Dario Valenzano** : [github](https://github.com/dvalenzano)
