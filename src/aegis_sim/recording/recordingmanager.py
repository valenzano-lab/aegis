"""Data recorder

Records data generated by the simulation.

When thinking about recording additional data, consider that there are three recording methods:
    I. Snapshots (record data from the population at a specific step)
    II. Flushes (collect data over time then flush)
    III. One-time records
    IV. Other: TE records
"""

import pathlib
import shutil
import logging

from aegis_sim import variables

from .terecorder import TERecorder
from .picklerecorder import PickleRecorder
from .popgenstatsrecorder import PopgenStatsRecorder
from .intervalrecorder import IntervalRecorder
from .flushrecorder import FlushRecorder
from .featherrecorder import FeatherRecorder
from .phenomaprecorder import PhenomapRecorder
from .summaryrecorder import SummaryRecorder
from .progressrecorder import ProgressRecorder
from .simpleprogressrecorder import SimpleProgressRecorder
from .popsizerecorder import PopsizeRecorder
from .ticker import Ticker
from .configrecorder import ConfigRecorder

# TODO write tests


class RecordingManager:
    """
    Container class for various recorders.
    Each recorder records a certain type of data.
    Most recorders record data as tables, except SummaryRecorder and PickleRecorder which record JSON files and pickles (a binary python format).
    Headers and indexes of all tabular files are explicitly recorded.

    -----
    GUI
    AEGIS records a lot of different data.
    In brief, AEGIS records
    genomic data (population-level allele frequencies and individual-level binary sequences) and
    phenotypic data (observed population-level phenotypes and intrinsic individual-level phenotypes),
    as well as
    derived demographic data (life, death and birth tables),
    population genetic data (e.g. effective population size, theta), and
    survival analysis data (TE / time-event tables).
    Furthermore, it records metadata (e.g. simulation log, processed configuration files) and python pickle files.

    Recorded data is distributed in multiple files.
    Almost all data are tabular, so each file is a table to which rows are appended as the simulation is running.
    The recording rates are frequencies at which rows are added; they are expressed in simulation steps.
    """

    def init(self, custom_config_path, overwrite):
        self.odir = self.make_odir(custom_config_path=custom_config_path, overwrite=overwrite)
        # TODO make subfolders

    def initialize_recorders(self, TICKER_RATE):
        self.terecorder = TERecorder(odir=self.odir)
        self.picklerecorder = PickleRecorder(odir=self.odir)
        self.popgenstatsrecorder = PopgenStatsRecorder(odir=self.odir)
        self.guirecorder = IntervalRecorder(odir=self.odir)
        self.flushrecorder = FlushRecorder(odir=self.odir)
        self.featherrecorder = FeatherRecorder(odir=self.odir)
        self.phenomaprecorder = PhenomapRecorder(odir=self.odir)
        self.summaryrecorder = SummaryRecorder(odir=self.odir)
        self.progressrecorder = ProgressRecorder(odir=self.odir)
        self.simpleprogressrecorder = SimpleProgressRecorder(odir=self.odir)
        self.ticker = Ticker(odir=self.odir, TICKER_RATE=TICKER_RATE)
        self.popsizerecorder = PopsizeRecorder(odir=self.odir)
        self.configrecorder = ConfigRecorder(odir=self.odir)

    #############
    # UTILITIES #
    #############

    @staticmethod
    def make_odir(custom_config_path, overwrite) -> pathlib.Path:
        output_path = custom_config_path.parent / custom_config_path.stem  # remove .yml
        is_occupied = output_path.exists() and output_path.is_dir()
        if is_occupied:
            if overwrite:
                shutil.rmtree(output_path)
            else:
                raise Exception(f"{output_path} already exists. To overwrite, add flag --overwrite or -o.")
        return output_path

    @staticmethod
    def make_subfolders(paths):
        # TODO relink paths that are now in classes
        for path in paths:
            path.mkdir(exist_ok=True, parents=True)

    def is_extinct(self) -> bool:
        if self.summaryrecorder.extinct:
            logging.info(f"Population went extinct (at step {variables.steps}).")
            return True
        return False
