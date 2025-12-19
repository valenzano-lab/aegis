# Origin tracking

This document describes the design and implementation of origin tracking in AEGIS. Origin tracking is useful when the simulation has population structure (i.e. there are multiple populations between individuals can migrate and mate) and we want to understand which parts of the genome originate from which population.

## How origin tracking works?
An array `origins` is saved in the `Population` object. It has the same dimensions of the `genomes` array; it is an integer array. At initialization, all the `origins` vectors for individuals in one population carry one value (e.g. 1), while all individuals in the other population carry another (e.g. 2). When two individuals sexually reproduce, their offspring will get some bits from one parent, and some from the other, depending on how the gametes recombine. As the genome for the offspring is assembled, so are the `origins` vectors recombined and saved.

### An example of two individuals reproducing and the resulting offspring

Parent A, initially from population 2
Genome:     001100111101
Origin:     222222222222

Parent B, initially from population 3
Genome:     101101001010
Origin:     333333333333

Resulting offspring
Genome:     001101001010
Origin:     222333333333

In another way, origin expressed using A and B:
Parent:     AAABBBBBBBBB
Note that the first three bits come from the parent A, while the rest of the bits come from the parent B. This means that there has been a recombination event between bit 3 and bit 4.

### An example of two individuals reproducing, one of which is already a hybrid
Parent A, hybrid
Genome:     001101001010
Origin:     222333333333

Parent B, non-hybrid
Genome:     110000101011
Origin:     222222222222

Resulting offspring
Genome:     001101001011
Origin:     222333333332

In another way, origin expressed using A and B:
Parent:     BBBBBBBBBBBA
Note that in this case only the last bit stems from parent B; however, there are other bits that originate from the population 2, but they come from parent A which is already a hybrid. The recombination event occurred between the ultimate and the penultimate bit.

## Recording

Origin information is recorded in feathers. It is important that the order of individuals in genomes feathers and origins feathers is the same so they can be cross-referenced.

## Tasks
- [ ] Add origins as a vector to the population.
- [ ] Origins is initialized as the population is initialized. There should be two modes: individual-specific and population-specific initialization.
- [ ] Origins is propagated during reproduction to offspring.
- [ ] During mutation, origins is preserved.
- [ ] Record in feathers. It is important that the order of individuals in genomes feathers and origins feathers is the same so they can be cross-referenced.