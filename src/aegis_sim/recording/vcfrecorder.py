import logging
import pathlib

from .recorder import Recorder
from aegis_sim import variables
from aegis_sim.parameterization import parametermanager
from aegis_sim.utilities.vcf import encode_population_to_vcf
from aegis_sim.utilities.funcs import skip


class VCFRecorder(Recorder):
    """Write the living population as a VCF (one row per genome bit, one column
    per individual) at the configured rate.

    Output: /vcf/step{step}.vcf
    Rate parameter: VCF_RATE (0 disables; still writes at the final step when
    VCF_RATE > 0).
    Only the composite architecture is supported.
    """

    def __init__(self, odir: pathlib.Path):
        self.odir = odir / "vcf"
        self.init_odir()

    def write(self, population):
        """
        # OUTPUT SPECIFICATION
        path: /vcf/step{step}.vcf
        filetype: vcf (v4.2)
        category: log
        description: Per-individual diploid genotypes at every genome bit, formatted as a self-contained VCF (header records architecture metadata for decoding). Compatible with PLINK, vcftools, ADMIXTOOLS, scikit-allel.
        trait granularity: bit (one row per logical bit position)
        time granularity: snapshot
        frequency parameter: VCF_RATE
        structure: VCF v4.2.
        """

        if parametermanager.parameters.VCF_RATE <= 0:
            return

        step = variables.steps
        should_skip = skip("VCF_RATE")
        is_first_step = step == 1
        is_last_step = step == parametermanager.parameters.STEPS_PER_SIMULATION

        if not (is_first_step or not should_skip or is_last_step):
            return

        if len(population) == 0:
            return

        logging.debug(f"vcf recorded at step {step}.")
        encode_population_to_vcf(
            population=population,
            output_dir=self.odir,
            name=f"step{step}",
        )
