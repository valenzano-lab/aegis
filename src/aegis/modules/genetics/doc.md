<!-- Last updated for version 2.x # TODO -->

<!-- when explaining, consider technical aspect (what technical problem is this solving), biological aspect (what biology is this simulating); include graphics and formulas -->

# Genomes and phenotypes
- why do we have them? why have pseudogenomes, why not directly phenotypes?
- genomes vs pseudogenomes vs genotypes
- what is the difference between intrinsic and total (refer to docstring in architect.py)


# Structure of pseudogenomes

Similarly to how genomes can be given as a simple array of bases (e.g. ATTTGATAC)
which is then further organized into a more complex structure of genes and chromosomes,
pseudogenomes in AEGIS can also be given as a simple array of bits (e.g. 100101001)
that can be organized in a more complex structure.

In AEGIS, genomes $g$ are three-dimensional arrays. That means that each site $s$ can be referred to
using a three-number index ($i,j,k$), i.e. $s_{i,j,k}$. Consider an example of a diploid individual.

```
(chromosome set 1)
0 1 1 1 0 1 1 0 (locus 1)
0 1 1 - 1 1 0 0 (locus 2)
1 0 1 0 0 1 1 0 (locus 3)
...
1 0 1 1 0 1 1 0 (locus L)

(chromosome set 2)
1 1 1 1 0 1 1 0 (locus 1)
0 1 0 1 0 1 = 0 (locus 2)
1 0 1 0 1 1 1 0 (locus 3)
...
1 0 0 1 0 1 1 0 (locus L)
```

This individual has two sets of chromosomes (diploid), which are further organized into loci containing eight bits.
The site marked as a hyphen `-` is $s_{1,2,4}$ because it is the site on the first chromosome set ($i=1$),
second locus ($j=2$) in the fourth position ($k=4$).
In another example, the site marked as the equal sign `=` is $s_{2,2,7}$.

### What is the purpose of using a three-number index versus a simple one-number index?

### Why do loci have eight bits?

The number of bits in a locus can be modified by the user. The parameter that sets that value is `BITS_PER_LOCUS`.
This setting is important for a __composite__ genetic architecture. Under a __modifying__ genetic architecture, this parameter is $1$; so for all sites $k$ is 1.

Overall, a higher `BITS_PER_LOCUS` increases the granularity of phenotypes (the value range that phenotypes can take on).
To understand better how `BITS_PER_LOCUS` affects a simulation, read more about the composite genetic architecture. # TODO insert better reference to cga


### Why are there chromosome sets?

The number of chromosome sets depends on the chosen ploidy. If the simulated population is haploid, there is only one chromosome set so for all sites $i$ is $1$.

### why is the structure the way it is? how could it be different?

# Computing phenotypes from pseudogenomes

The computation process occurs in four steps:
1. env
2. ploidy
3. composite vs modifying
4. low and high bounds


## Env

## Ploidy

## ...

### Composite
$ phenotype(age, trait) = \sum_{i=0}^{BPL}(haplogenotype[age, i] * weight(i))$

### Modifying

## Low and high bounds

why not

# Phenotype structure
