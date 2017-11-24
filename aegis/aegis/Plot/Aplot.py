########################################################################
# AEGIS - Ageing of Evolving Genomes in Silico                         #
# Module: Plot                                                         #
# Classes: APlot [better name tbc]                                     #
# Description: Wrapper object that takes and stores a passed Record    #
#   object and implements plotting methods on it.                      #
########################################################################

class APlot:
    """Wrapper class for storing a Record and its associated plots."""

    def __init__(self, record):
        """Import a Record object and initialise plotting methods."""
        self.record = record
