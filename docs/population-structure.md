# Population structure

This document describes the design and implementation of population structure in AEGIS. 

## What is population structure?
Under default configuration, all individuals live in a single environment, as a part of one population; individuals share the same resources, they are exposed to same environmental hazards, disease reservoir, and predators; they also reproduce within the population without reproductive preference.

Population structure enables modeling two or more populations in which these factors can be independent. The populations still interact via individuals that migrate between the populations.

## Initialization

## Pickling

## Recording

## Tasks
- [ ] Enable multiple bioreactors.
- [ ] Enable migration between bioreactors.
- [ ] Add migration rate as a parameter.
- [ ] Handle initialization for the multiple bioreactors.
- [ ] Handle pickling.
- [ ] Recording?
- [ ] Will this interfere with some population-specific behaviors, e.g. timing of hatching events?