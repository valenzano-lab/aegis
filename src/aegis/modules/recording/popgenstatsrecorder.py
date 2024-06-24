import numpy as np

from aegis.hermes import hermes


class PopgenStatsRecorder:
    def __init__(self, odir):
        self.odir = odir / "popgen"

    def write(self, genomes, mutation_rates):
        """
        Record population size in popgenstats, and record popgen statistics

        # OUTPUT SPECIFICATION
        format: csv
        content: population genetic stats
        dtype:
        index:
        header:
        column:
        rows: 
        """
        hermes.modules.popgenstats.record_pop_size_history(genomes.array)

        if hermes.skip("POPGENSTATS_RATE") or len(genomes) == 0:
            return

        hermes.modules.popgenstats.calc(genomes.array, mutation_rates)

        # Record simple statistics
        array = list(hermes.modules.popgenstats.emit_simple().values())
        if None in array:
            return

        with open(self.odir / "simple.csv", "ab") as f:
            np.savetxt(f, [array], delimiter=",", fmt="%1.3e")

        # Record complex statistics
        complex_statistics = hermes.modules.popgenstats.emit_complex()
        for key, array in complex_statistics.items():
            with open(self.odir / f"{key}.csv", "ab") as f:
                np.savetxt(f, [array], delimiter=",", fmt="%1.3e")
