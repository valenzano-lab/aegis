import logging
import pathlib

from .recorder import Recorder
from aegis_sim import variables, submodels
from aegis_sim.parameterization import parametermanager


HEADER = "step,n_alive,n_alleles,n_carriers,allele_freq\n"


class SelectionRecorder(Recorder):
    """Track the allele frequency at the injection locus over time.

    Active only when ALLELE_INJECTION_STEP > 0. Records start at step 1
    (so the baseline before injection is visible) and continue through
    the rest of the simulation.

    Output: /selection/selection.csv with columns
        step,n_alive,n_alleles,n_carriers,allele_freq
    where n_alleles = 2 * n_alive (diploid count of alleles at the locus),
    n_carriers = count of chromatids carrying ALLELE_INJECTION_ALLELE at
    the injection locus, allele_freq = n_carriers / n_alleles.
    """

    def __init__(self, odir: pathlib.Path):
        self.odir = odir / "selection"
        self.init_odir()
        self._file_path = self.odir / "selection.csv"
        self._file = None
        # Cached physical locus + bit, populated on first write.
        self._physical_locus = None
        self._bit_in_locus = None
        self._tracked_allele = None

    def _resolve_locus(self):
        if self._physical_locus is not None:
            return True
        from aegis_sim import parameterization

        trait_name = parametermanager.parameters.ALLELE_INJECTION_TRAIT
        age = int(parametermanager.parameters.ALLELE_INJECTION_AGE)
        bit_in_locus = int(parametermanager.parameters.ALLELE_INJECTION_BIT)
        allele = bool(int(parametermanager.parameters.ALLELE_INJECTION_ALLELE))

        trait = parameterization.traits.get(trait_name)
        if trait is None or trait.length == 0:
            return False
        if trait.agespecific is True:
            if not (0 <= age < trait.length):
                return False
            logical_locus = trait.start + age
        else:
            logical_locus = trait.start

        self._physical_locus = int(submodels.architect.architecture.locus_permutation[logical_locus])
        self._bit_in_locus = bit_in_locus
        self._tracked_allele = allele
        return True

    def _ensure_open(self):
        if self._file is not None:
            return
        mode = "a" if self._file_path.exists() and self._file_path.stat().st_size > 0 else "w"
        self._file = open(self._file_path, mode)
        if mode == "w":
            self._file.write(HEADER)

    def write(self, population):
        if parametermanager.parameters.ALLELE_INJECTION_STEP <= 0:
            return
        if not self._resolve_locus():
            return
        n = len(population)
        if n == 0:
            return

        bits = population.genomes.array[:, :, self._physical_locus, self._bit_in_locus]
        # bits shape: (n, 2)
        n_alleles = bits.size  # n * 2
        n_carriers = int((bits == self._tracked_allele).sum())
        freq = n_carriers / n_alleles if n_alleles else 0.0

        self._ensure_open()
        self._file.write(f"{variables.steps},{n},{n_alleles},{n_carriers},{freq:.6f}\n")
        self._file.flush()
