[![PyPI version](https://badge.fury.io/py/aegis-sim.svg)](https://badge.fury.io/py/aegis-sim)
[![Python 3.6+](https://img.shields.io/badge/python-3.6%2B-blue)](https://www.python.org/downloads/release/python-360/)

# AEGIS

> Aging of Evolving Genomes In Silico (AY-jis, /eɪd͡ʒɪs/)

Numerical model for life history evolution of age-structured populations under customizable ecological scenarios.

<!-- TODO describe what aegis is for and whom is it for -->

## How to use

You can run AEGIS simulations on a webserver or locally. The webserver is especially useful if you want to try AEGIS out and run a couple of simple simulations. For more demanding simulations, it is best to install and run AEGIS on your local machine.

### Webserver use

You can access the AEGIS webserver [here](). The server is running AEGIS GUI.<!-- TODO update link -->

### Local use

You can install AEGIS locally using pip (`pip install aegis-sim`). The package is available on https://pypi.org/project/aegis-sim/. You can use AEGIS with a GUI or in a terminal. GUI is useful for running individual simulations, while the terminal is useful for running batches of simulations.

```bash
aegis gui # starts GUI
aegis sim -c {path/to/config_file} # runs a simulation within a terminal
aegis --help # shows help documentation
```

To run simulations within a terminal, you need to prepare config files in [YAML](https://en.wikipedia.org/wiki/YAML) format
which contain custom values for simulation parameters. The list of parameters, including their descriptions and default values you can find [here](src/aegis/documentation/dynamic/default_parameters.md).
An example of a config file:

```yml
RANDOM_SEED: 42
STEPS_PER_SIMULATION: 10000
AGE_LIMIT: 50
```

### Developer installation

If you want to contribute to the codebase, install AEGIS from github:

```bash
python3 -m pip install -e git+https://github.com/valenzano-lab/aegis.git#egg=aegis-sim
```

<!-- or
```bash
git clone git@github.com:valenzano-lab/aegis.git
cd aegis
make install_dev
``` -->

If you are having installation issues, check that pip is up to date (`python3 -m pip install --upgrade pip`).

<!-- TODO update install_dev script -->

### AEGIS GUI
Graphical user interface for AEGIS can be used on the webserver or with a local installation. It contains sections for launching and analyzing/plotting simulations.

<img width="418" alt="Screenshot_21" src="https://github.com/user-attachments/assets/520bb69c-a1cd-404a-bdf8-a541340e699a" style="border-radius: 10px;">


## Documentation

### Model description

Most documentation about the model is available within the GUI itself, including description of [inputs](src/aegis/documentation/dynamic/default_parameters.md), [outputs](src/aegis/documentation/dynamic/output_specifications.md), [submodels](src/aegis/documentation/dynamic/submodel_specifications.md) and the [genetic architecture](src/aegis_sim/submodels/genetics/doc.md). Use the [webserver]() or a local installation to access the GUI. <!-- TODO update link --> Further information is available in the following articles:

- [AEGIS: An In Silico Tool to model Genome Evolution in Age-Structured Populations (2019)](https://www.biorxiv.org/content/10.1101/646877v1)
- [An In Silico Model to Simulate the Evolution of Biological Aging (2016)](https://www.biorxiv.org/content/10.1101/037952v1)
<!-- TODO including ODD as modeled by https://www.jasss.org/23/2/7.html-->

### API reference

Exhaustive, searchable API reference made by pdoc is available [here](https://valenzano-lab.github.io/aegis/).

## Contributors

- **Martin Bagić** (v2): [email](martin.bagic@outlook.com), [github](https://github.com/martinbagic)
- **Dario Valenzano** (v1, v2): [github](https://github.com/dvalenzano)
- **Erik Boelen Theile** (v2)
- **Arian Šajina** (v1): [github](https://github.com/ariansajina)
- **William Bradshaw** (v1): [github](https://github.com/willbradshaw)
