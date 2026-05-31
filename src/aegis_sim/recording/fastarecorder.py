import logging
import pathlib

from .recorder import Recorder
from aegis_sim import variables
from aegis_sim.parameterization import parametermanager
from aegis_sim.utilities.fasta import encode_population_to_fasta
from aegis_sim.utilities.funcs import skip


class FastaRecorder(Recorder):
    """Write the living population as a FASTA reference (one record per individual)
    plus a sidecar JSON mapping for the round-trip decoder.

    Output: /fasta/step{step}.genome.fasta + /fasta/step{step}.mapping.json
    Rate parameter: FASTA_RATE (0 disables; still writes at the final step if a
    non-zero rate is set or the rate is 0).
    """

    def __init__(self, odir: pathlib.Path):
        self.odir = odir / "fasta"
        self.init_odir()

    def write(self, population):
        """
        # OUTPUT SPECIFICATION
        path: /fasta/step{step}.genome.fasta and /fasta/step{step}.mapping.json
        filetype: fasta + json
        category: log
        description: Per-individual genome FASTA (4-letter XOR-masked packing) and sidecar mapping for lossless decode back to the original bit array and architect-derived phenotypes.
        trait granularity: individual
        time granularity: snapshot
        frequency parameter: FASTA_RATE
        structure: FASTA (one record per individual) + JSON sidecar (encoding metadata + XOR mask).
        """

        if parametermanager.parameters.FASTA_RATE <= 0:
            return

        step = variables.steps
        should_skip = skip("FASTA_RATE")
        is_first_step = step == 1
        is_last_step = step == parametermanager.parameters.STEPS_PER_SIMULATION

        if not (is_first_step or not should_skip or is_last_step):
            return

        if len(population) == 0:
            return

        mask_seed = parametermanager.parameters.FASTA_MASK_SEED

        logging.debug(f"fasta recorded at step {step}.")
        encode_population_to_fasta(
            population=population,
            output_dir=self.odir,
            name=f"step{step}",
            mask_seed=mask_seed,
        )
