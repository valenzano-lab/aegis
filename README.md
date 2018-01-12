![Logo of the project](https://raw.githubusercontent.com/jehna/readme-best-practices/master/sample-logo.png)

TODO We need a cool logo!

# AEGIS
> Ageing of Evolving Genomes In Silico

A highly versatile numerical model of genome evolution - both sexual and asexual - 
for a population of agents whose fitness parameters are encoded in a bit arrays
which are free to evolve due to mutation and recombination.

## Installation

### currently
First clone the repo and run the tests:
```bash
git clone git@bitbucket.org:willbradshaw/genome-simulation.git
python setup.py test
```
If all tests pass, install aegis via pip in editable mode:
```bash
pip install -e .
```

### once project on PyPI
```bash
pip install aegis
```
## Developing
To be able to inspect, edit or expand the code, do the following:

```bash
git clone git@bitbucket.org:willbradshaw/genome-simulation.git
python setup.py test
pip install -e .
```
### Contributing
If you'd like to contribute, please fork the repository and use a feature
branch. Pull requests are warmly welcome.

### Conventions
If you wish to expand the model's functionality, make a copy of the Core module,
rename it and edit as desired.
Sticking to this modular practice allows us to add modules with which we can
investigate different questions of evolutionary biology without sacrificing
functionality attained so far.

Other practices we work by:
* [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html)
* [Git housekeeping](https://railsware.com/blog/2014/08/11/git-housekeeping-tutorial-clean-up-outdated-branches-in-local-and-remote-repositories/")
* [pytest](https://docs.pytest.org/en/latest/) testing

#### Helpful material for developing
This is a schematic representation of aegis class hierarchy in the Core module:

![aegis class hierarchy](./ach.png)

## Features
It is highly advised to read **this publication** to better understand both the 
conceptual and the practical aspect of the model. (TODO **this publication** links
to publication explaining the model)
For a quick dive into the usage of aegis reading this README and inspecting the 
*configuration file* (see below) should suffice.

## Usage
This is the help output of aegis:
```
usage: aegis [-h] [-m <str>] [-v] [-r <int>] [-p <str>] script infile

Run the genome ageing simulation.

positional arguments:
  script                AEGIS script to run:
                            run - run the simulation with the config file specified in
                                  infile
                            plot - plot data from record object specified in infile
                            getconfig - copies the default config file to infile
  infile                path to input config file

optional arguments:
  -h, --help            show this help message and exit
  -m <str>, --module <str>
                        AEGIS module to use for simulation run (default: 'Core')
  -v, --verbose         display full information at each report stage
                            (default: only starting population)
  -r <int>              report information every <int> stages (default: 100)
  -p <str>, --profile <str>
                        profile genome simulation with cProfile and save to given path
```
The *configuration file* is instrumental to running simulations with aegis.
You would copy a default configuration file to my_config.py in your cwd like so:
```bash
aegis getconfig ./my_config.py
```
This file defines *all* simulation parameters. Edit the parameter values as desired
with your favorite text editor and then run the simulation:
```bash
vim my_config.py
aegis run my_config.py
```
You will see informative output being written in the standard output as the
simulation progresses and will be informed when the simulation has ended.

If you wished to see full information displayed every 20 stages you would have
written:
```bash
aegis run my_config -v -r 20
```
Once simulation has finished, a directory will have been written in your cwd.
In *my_config.py* you had to designate what files will be saved at completion.
This could have been:
* 0: records only
* 1: records and final populations
* 2: records and final populations and all snapshot populations

Also you had to designate the prefix of the output directory; say `sim1`,
then your files are saved in  `./sim1_files`.

To plot the data from the from the first run you would do:
```bash
aegis plot sim1_files/records/run0.rec
```
The plot are saved in `./sim1_plots`.
### Examples
Following are example configuration files for different simulation scenarios with
explanations.

#### Scenario 1
TODO Link configuration file and explain them.

#### Scenario 2
TODO Link configuration file and explain them.

#### Scenario 3
TODO Link configuration file and explain them.

#### Scenario 4
TODO Link configuration file and explain them.

## Related publications
[An In Silico Model to Simulate the Evolution of Biological Aging](www.biorxiv.org/content/early/2016/01/26/037952)

## Team
* Dario Valenzano   Dario.Valenzano@age.mpg.de
* William Bradshaw  William.Bradshaw@age.mpg.de
* Arian Šajina      Arian.Sajina@age.mpg.de

## Licensing
The code in this project is licensed under MIT license.

## Acknowledgments
This project is developed in the [Valenzano Lab](http://valenzano-lab.age.mpg.de) of
the [Max Planck Institute for Biology of Ageing, Cologne](https://www.age.mpg.de).
We thank all the lab members and friends of the lab for their constructive
comments and suggestions.
