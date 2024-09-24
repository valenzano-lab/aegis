<!-- Last updated for version 2.x # TODO -->

<!-- when explaining, consider technical aspect (what technical problem is this solving), biological aspect (what biology is this simulating); include graphics and formulas -->

In AEGIS, agents have heritable traits that are inherited by their offspring, in a mutated state.
Fertility, intrinsic mortality, growth rate and mutation rate can be set to be heritable but they don't have to be heritable; they can be set to be equal across individuals.

### Why use pseudogenomes?

To encode heritable traits, AEGIS uses pseudogenomes – bitstrings whose sites translate into phenotypic effects. Though intrinsic traits could be directly saved by the simulation (as many agent-based models do) as real-number vectors, pseudogenomes have multiple uses, as records of mutations. First, they enable recombination – a process central to evolution under sexual reproduction – as recombination requires maintenance of information about mutation sets. Second, they enable analysis of evolutionary dynamics and population genetics; e.g. mutation trajectories, selection patterns, diversity metrics.

### Genomes vs pseudogenomes vs genotypes

In AEGIS, the term genome is most often used and broadly overlaps with the biological concept of genome.
In cases where the distinction between the biorealistic and the computational genome is important, the term pseudogenome is used to remind that the computational genome is different from the empirical one.
As is the case in genetics, the term genotype refers to a particular instance of a genome (i.e. a particular sequence of bases or bits).

### Intrinsic vs observed phenotypes (capacities vs outcomes)

Observed phenotypes are results of genetics and the environment. Genetics encodes hidden capacities (intrinsic phenotypes, e.g. the ability to produce offspring) while the observed phenotype refers to specific outcomes (e.g. having produced an offspring or not having produced an offspring). The observed phenotype depends on genetics (e.g. by definition, a more intrinsically fertile individual has higher chances of producing offspring) but also on the social and abiotic environment (e.g. a highly fertile individual might not produce an offspring because it did not find a mate or there was not enough food). The observed phenotype is stochastic – it is random in nature (subject to chance, unpredictable on an individual basis; e.g. the outcome of a single reproductive attempt is unpredictable); however, its outcomes are driven by particular tendencies (predictable on an aggregate basis, e.g. the overall outcome of many reproductive attempts).

In AEGIS, intrinsic phenotypes are also called capacities, and hidden or unobservable phenotypes. Observed phenotypes are also called realized, total or final phenotypes; or outcomes when referring to specific phenotypic events (e.g. a reproductive attempt).

For example, there are intrinsic mutation rates and observed mutation rates. When an individual reproduces, an offspring is created with identical genetics which undergo mutation. An intrinsic mutation rate (which can be individual-specific) determines the probability of each genomic site to flip (from 0 to 1 or vice versa); however, the realized mutation rate (the number of mutations divided by number of mutable sites) can be quite different from the intrinsic rate. The realized rate is stochastic and it will follow a binomial distribution.

# Structure of pseudogenomes

Similarly to how genomes can be given as a simple array of bases (e.g. ATTTGATAC)
which is then further organized into a more complex structure of genes and chromosomes,
pseudogenomes in AEGIS can also be given as a simple array of bits (e.g. 100101001)
that can be organized in a more complex structure.

In AEGIS, genomes $g$ are three-dimensional arrays. That means that each site $g$ can be referred to
using a three-number index ($i,j,k$), i.e. $g_{i,j,k}$. Consider an example of a diploid individual.

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
The site marked as a hyphen `-` is $g_{1,2,4}$ because it is the site on the first chromosome set ($i=1$),
second locus ($j=2$) in the fourth position ($k=4$).
In another example, the site marked as the equal sign `=` is $g_{2,2,7}$.

##### The rationale behind the three-dimensionality of pseudogenomes

This structure is biologically and computationally motivated. Biological motivation stems from the need to define chromosome sets (because they undergo specific biological processes such as recombination, assortment and dominance) and usefulness of loci (as they allude to genes – sequences of bases that work together to target a specific phenotypic trait). Computational motivation results from the biological motivation – e.g. computing recombination works different than computing phenotypic effects of a locus; thus organizing the genome into multiple dimensions aids with writing computational algorithms.

##### What is a locus? Why do loci have eight bits?

Locus is a sequence of multiple bases that all contribute to the same phenotypic trait (e.g. the fertility rate). This means that when one bases is mutated, the phenotypic trait will be modified (fertility rate increased or decreased). The same is true for empirical genes.

In the sketches above, each locus has eight bits; however, that number is modifiable by the user. The parameter that sets that value is `BITS_PER_LOCUS`. It is not important that the number of bases is similar to number of bases in real genes. This setting is important for a **composite** genetic architecture. Under a **modifying** genetic architecture, this parameter is $1$; so for all sites $k$ is 1.

Overall, a higher `BITS_PER_LOCUS` increases the granularity of phenotypes (the value range that phenotypes can take on).
To understand better how `BITS_PER_LOCUS` affects a simulation, read more about the composite genetic architecture.

<!-- # TODO include __composite__ architecture info -->

# Structure of intrinsic phenotypes

AEGIS can encode four intrinsic traits – intrinsic mortality, intrinsic fertility, intrinsic mutation rate and intrinsic growth rate. Intrinsic traits are individual-specific, heritable and evolvable. Depending on the simulation setup, some of these traits can be made fixed (thus not intrinsic anymore).

When referring to intrinsic phenotypes, we use the following notation: $m^i_a$, $f^i_a$, $mut^i_a$, $grw^i_a$ to refer to intrinsic mortality, fertility, mutation rate and growth rate of an individual $i$ at the age of $a$.

# Genotype-phenotype map (GPM)

Converting a genotype to a phenotype requires multiple steps. When explaining each step, we will state the biological rationale (what biological process is modeled by this computation) or the technical rational (what computational requirement is this fulfilling) and the computational formula and/or description.

The output of each step ($o^{step}$) is taken as input for the next step. There are four steps:

1. Environmental drift
2. Ploidy
3. Genetic architecture
   - _composite_
   - _modifying_
4. Phenotypic bounds

#### 1. Environmental drift

##### Motivation

This step is important in simulating environmental drift; i.e. a non-cyclic, progressively changing environment. When environment drifts, the fitness of all phenotypes shifts as they become more or less suited for the novel environment. Shifting environments are important for studying the effects of long-term environmental change (e.g. climate change, resource depletion) on evolution. Furthermore, it enables indefinite adaptive evolution because as the environment shifts, the population has to evolve to adapt to it. In stable environments, populations cease to adapt once they come close to the fitness peak.

##### Computation

$$g \oplus \epsilon \rightarrow o^1_{i,j,k}$$

##### Explanation

The relevant parameter here is _environmental drift rate_, $ER$. When $ER=0$, there is no environmental drift (the environment is stable); when it is non-zero, the environment drifts; $ER$ signifies how often the environment drifts (how many steps does it take for environment to drift).

The state of the environment is saved in the array $\epsilon$ which has the same structure as a genome ($g$). In the beginning of the simulation, $\epsilon$ only contains zeros but every $ER$ steps, a random bit in $\epsilon$ flips.

Note that $\oplus$ is a XOR operation which acts as a flip; i.e.
$$g_{i,j,k} \oplus \epsilon_{i,j,k} \rightarrow o^1_{i,j,k} \text{ when } \epsilon_{i,j,k} = 0$$
$$g_{i,j,k} \oplus \epsilon_{i,j,k} \rightarrow \neg\ o^1_{i,j,k} \text{ when } \epsilon_{i,j,k} = 1.$$

When $\epsilon$ is 1, this computation flips the homologous genomic bit, and when $\epsilon$ is 0, the computation keeps the value of the homologous genomic bit.

#### 2. Ploidy

##### Motivation

This step is important in handling diploidy and simulating inheritance patterns (e.g. dominance).

##### Computation

$$o^1_{i,j,0} = o^1_{i,j,1} \rightarrow o^2_{i,j} := o^1_{i,j,0}$$
$$o^1_{i,j,0} \ne o^1_{i,j,1} \rightarrow o^2_{i,j} := DF$$

##### Explanation

The relevant parameter here is the _dominance factor_, $DF$. When homologous sites are homozygous, the combined value is the value of either site ($1$ if both chromosome sets have a $1$, or $0$ when both have a $0$). When homologous sites are heterozygous, the combined value is $DF$.

The value of $DF$ will determine the inheritance pattern:

| DF | pattern |
| -------- | ------- |
| 0 | recessive |
| 0-1 | partial dominant |
| 0.5 | true additive |
| 1 | dominant |
| 1+ | overdominant |

#### 3. Genetic architecture

##### Motivation

Similar to how genes have effects on phenotypic traits as well as individual sites, so do loci and individual sites in pseudogenomes. AEGIS encodes no molecular information, so no individual, specific genes are modeled together with the information on how they affect which phenotypic traits. Instead, AEGIS uses genotype-phenotype mapping – a computational method that translates the state of each pseudogenomic site into a phenotypic effect. The final phenotype, then, is the aggregate of all those individual phenotypic effects.

Genotype-phenotype mapping requires as input a genetic architecture, $GA$.
In AEGIS, there are two kinds of architectures – a _composite_ and a _modifying_ architecture. When compared to a _modifying_ architecture, a _composite_ architecture is fast, simple to define, easy to plot and understand; however, it cannot simulate sites with age-pleiotropic or trait-pleiotropic effects (i.e. a genetic site that affects two traits or affects a single trait at two different ages). The _modifying_ architecture can simulate pleiotropy (age-pleiotropy and/or trait-pleiotropy). An additional advantage of the _modifying_ architecture is that linkage between phenotypically related genes is low; i.e. genes that affect a specific trait at a specific age will usually not find itself next to another gene that affects the same trait at a similar age.

##### Computation (general)

We can resolve the genetic architecture as a weighted sum

$$o^3_{t,a} \leftarrow \sum_{i,j} o^2_{i,j} * GA_{i,j,t,a}$$

or as a matrix multiplication

$$GA \times o^2 \rightarrow o^3$$

where the intermediate pseudogenome $o^2$ is flattened (to one dimensions) and the $GA$ is flattened to have $`_{i,j}`$ on one axis (as $o^2$) and $`_{t,a}`$ on the other.

Conceptually, it is simpler to think of it as a weighted sum. $GA_{i,j,t,a}$ are user-set inputs, the weights of the weighted sum. When $o^2_{i,j}$ is greater than 0, it contributes to a trait $t$ at an age $a$ with the weight of $GA_{i,j,t,a}$.

Note that $GA$ is only a concept. The user does not enter individual weights of the genetic architecture; furthermore, the exact computation procedure depends on the type of the genetic architecture used. See below.

###### Computation for composite architecture

Under composite architecture, each locus is responsible for setting a phenotypic value of one trait at one age. Each locus has a user-defined number of bits, $BPL$ (parameter `BITS_PER_LOCUS`). Each bit has a weight, determined by the locus architecture $LA_i$. Under the simplest locus architecture, every bit has the same weight; i.e. $LA_{i} = \frac{1}{BPL}$. More complex locus architectures can be made where bits are increasingly more important within a locus.

<!-- # TODO use := instead of arrows -->

$$o^{2*}\_{j} \leftarrow \sum_{i=0}^{BPL} o^{2}_{i,j} \cdot LA_i$$

The sequence $o^{2*}_j$ contains locus values. These are now mapped onto various traits. Per code convention, traits are encoded in the following sequence – intrinsic mortality, intrinsic fertility, intrinsic mutation rate and intrinsic growth rate.

$$
o^{3}_{mortality, a} := o^{2*}\_{a}
$$

$$
o^{3}_{fertility, a} := o^{2*}\_{a+\Delta A}
$$

$$
o^{3}_{mutation\ rate, a} := o^{2*}\_{a+2\Delta A}
$$

$$
o^{3}_{growth\ rate, a} := o^{2*}\_{a+3\Delta A}
$$

Factor $\Delta A$ is the parameter `AGE_LIMIT`, the maximum age that can be obtained by any individual within a custom simulation.
These formulas simply mean that the phenotypic values for mortality are encoded in the first $\Delta A$ loci, values for fertility in the second $\Delta A$ loci, etc.

###### Computation for modifying architecture

Under modifying architecture, the computation is simpler. `BITS_PER_LOCUS` is set to 1, so $o^2_{i,j} = o^2_{i,0} \rightarrow o^{2*}_i.$

The weighted sum formula then becomes
$o^3_{t,a} \leftarrow \sum_{i} o^{2*}_{i} * GA\_{i,t,a}$
which means that setting $GA$ becomes simpler.

Weights of the genetic architecture ($GA_{i,t,a}$) are sampled from user-specified distributions.

<!-- # TODO explain how to set GA -->

#### 4. Phenotypic bounds

##### Motivation

In case probability traits (intrinsic survival, reproduction and mutation rates) compute to values over 1 or under 0, they are cut back to the valid interval of [0,1].

##### Computation

$$o^4*{t,a} \leftarrow 1 \text{ when } o^3*{t,a} > 1 $$
$$o^4*{t,a} \leftarrow 0 \text{ when } o^3*{t,a} < 0 $$
$$o^4*{t,a} \leftarrow o^3*{t,a} \text{ otherwise} $$

Note that these are equivalent:

$$o^4_{mortality, a} = m_{a}$$

$$o^4_{fertility, a} = f_{a}$$

$$o^4_{mutation\ rate, a} = mut_{a}$$

$$o^4_{growth\ rate, a} = grw_{a}$$

<!-- ### What is the rationale behind a three-number index (versus a simpler two-number of one-number index)?

The indexing structure is a conceptual tool – it is useful to think about the pseudogenome as a three-dimensional array since the different dimensions encode different meanings.

For example, the third dimension encodes the position of the chromosome set. Having an additional number in an index not only helps to more easily see that two sites are homologous (e.g. $g_{28,j,1}$ and $g_{28,j,2}$ are obviously homologous while $g_{28}$ and $g_{128}$ can only be determined to be homologous if we know that the pseudognome seize is $100$) – it also indicates that when we convert a pseudogenome

For example, when simulating sexual reproduction of diploid individuals, the pseudogenome will contain two chromosomal sets. To refer to a homologous site using a single-number index, one might have to say e.g. $g_{28}$ and $g_{128}$ for a pseudogenome of size $100$. Using a two-number structure, the indices would be $g_{28,1}$ and $g_{28,2}$.

The indexing structure is also a computational tool. When pseudogenomes are stored in AEGIS, they are stored as three-dimensional arrays. This structuring simplifies running vectorized operations along particular dimensions, which speed up computation. -->
