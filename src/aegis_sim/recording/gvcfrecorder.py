import logging
import pathlib

from .recorder import Recorder
from aegis_sim import variables
from aegis_sim.parameterization import parametermanager
from aegis_sim.utilities.gvcf import encode_population_to_gvcf
from aegis_sim.utilities.funcs import skip


class GVCFRecorder(Recorder):
    """Write the living population as a FASTA-coordinate multi-sample gVCF at
    the configured rate. Output is Clair3-format-compatible: ref blocks for
    invariant runs, <NON_REF> symbolic ALT, multi-allelic genotypes, synthetic
    GQ=99/DP=30 values. Drops straight into GLnexus joint-genotyping.

    Output: /gvcf/step{step}.gvcf
    Rate parameter: GVCF_RATE.
    """

    def __init__(self, odir: pathlib.Path):
        self.odir = odir / "gvcf"
        self.init_odir()

    def write(self, population):
        """
        # OUTPUT SPECIFICATION
        path: /gvcf/step{step}.gvcf
        filetype: gvcf (VCFv4.2 with <NON_REF> + reference blocks)
        category: log
        description: Multi-sample gVCF in FASTA-coordinate base positions. REF = consensus across the population, ALT = observed non-REF alleles + <NON_REF>. Designed as a drop-in replacement for Clair3 output so the existing GLnexus -> ABBA-BABA pipeline can use AEGIS truth as ground-truth comparison against read-derived calls.
        trait granularity: base position (one row per FASTA base or one row per reference block)
        time granularity: snapshot
        frequency parameter: GVCF_RATE
        structure: VCFv4.2 with gVCF extensions.
        """
        if parametermanager.parameters.GVCF_RATE <= 0:
            return

        step = variables.steps
        should_skip = skip("GVCF_RATE")
        is_first_step = step == 1
        is_last_step = step == parametermanager.parameters.STEPS_PER_SIMULATION

        if not (is_first_step or not should_skip or is_last_step):
            return

        if len(population) == 0:
            return

        logging.debug(f"gvcf recorded at step {step}.")
        encode_population_to_gvcf(
            population=population,
            output_dir=self.odir,
            name=f"step{step}",
        )
