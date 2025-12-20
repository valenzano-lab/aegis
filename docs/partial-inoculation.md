# Co-inoculation

This document describes the design and implementation of population co-inoculation in AEGIS.

## What is co-inoculation for?

Co-inoculation allows users to initialize new simulations using pre-evolved populations from multiple previous simulations in customizable proportions.

## How to use co-inoculation?

To use co-inoculation, specify the paths to the pre-evolved populations (e.g. `/{sim_directory}/pickles/{sim_step}`) and the proportions of the population to be used.

## What does co-inoculation do?

When co-inoculation is specified, AEGIS will load the specified pre-evolved populations and randomly sample individuals from each population according to the specified proportions to create the initial population for the new simulation.

The sampled individuals carry all phenotypic characteristics into the new simulation (including genetics, age, etc.), except `origin` which is reset; and can be individual-specific (i.e. every individual gets a different origin value), or pickle-specific (i.e. all individuals from one pre-evolved population get the same origin value, but different from individuals from another pre-evolved population).