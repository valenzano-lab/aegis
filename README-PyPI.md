# AEGIS
> Ageing of Evolving Genomes In Silico

A highly versatile discrete numerical model of genome evolution - both sexual and asexual - for a population of agents whose fitness parameters are encoded age-specifically in bit arrays which are free to evolve due to mutation and recombination.

This is a software implementation of the model described in this [article](https://www.biorxiv.org/content/early/2016/01/26/037952) in section "The Model".

## Who uses AEGIS?
* [Valenzano Lab](http://valenzano-lab.age.mpg.de/)

## Features
AEGIS can:
* simulate genome evolution in age-structured populations under a variety of evolutionary constraints
* simulate both asexually and sexually reproducing populations
* output simulation objects using [pickle](https://docs.python.org/2/library/pickle.html)
* output recorded statistics to a [csv](https://en.wikipedia.org/wiki/Comma-separated_values)
* generate figures from recorded statistics
* run a simulation until the population has reached evolutionary equilibrium (i.e. the genetic constitution is not expected to change anymore)

## Installation
Since aegis has dependencies, you might want to put the installation in an isolated Python environment with [virtualenv](https://virtualenv.pypa.io/en/stable/).
To install just do:
```shell
pip install mpi-age-aegis
```

## Usage
A detailed usage tutorial with examples is provided on our [GitHub page](https://github.com/valenzano-lab/aegis).

## Related articles
* [An In Silico Model to Simulate the Evolution of Biological Aging](https://www.biorxiv.org/content/early/2016/01/26/037952)

## Team
* **Arian Šajina**      (Arian.Sajina@age.mpg.de)
* **William Bradshaw**  (William.Bradshaw@age.mpg.de)
* **Dario Valenzano**   (Dario.Valenzano@age.mpg.de)

## Licensing
This project is licensed under MIT license.

## Acknowledgments
This project is developed in the [Valenzano Lab](http://valenzano-lab.age.mpg.de) of
the [Max Planck Institute for Biology of Ageing, Cologne](https://www.age.mpg.de).
We thank all the lab members and friends of the lab for their constructive
comments and suggestions.
