from aegis_sim.recording.featherrecorder import FeatherRecorder
from aegis_sim.recording.flushrecorder import FlushRecorder
from aegis_sim.recording.phenomaprecorder import PhenomapRecorder
from aegis_sim.recording.picklerecorder import PickleRecorder
from aegis_sim.recording.popgenstatsrecorder import PopgenStatsRecorder
from aegis_sim.recording.progressrecorder import ProgressRecorder
from aegis_sim.recording.summaryrecorder import SummaryRecorder
from aegis_sim.recording.terecorder import TERecorder
from aegis_sim.recording.ticker import Ticker
from aegis_sim.recording.intervalrecorder import IntervalRecorder


def extract_output_specification_from_docstring(method):
    """Extract information about the output file created by the method"""
    docstring = method.__doc__
    texts = docstring.split("# OUTPUT SPECIFICATION")[1:]
    for text in texts:
        parsed = {}
        for pair in text.strip().split("\n"):
            k, v = pair.split(":", maxsplit=1)
            parsed[k.strip()] = v.strip()
        yield parsed


output_specifications = [
    specification
    for method in (
        FeatherRecorder.write_genotypes,
        FeatherRecorder.write_phenotypes,
        FeatherRecorder.write_demography,
        FlushRecorder.write_age_at,
        PhenomapRecorder.write,
        PickleRecorder.write,
        PopgenStatsRecorder.write,
        ProgressRecorder.write_to_progress_log,
        SummaryRecorder.write_input_summary,
        SummaryRecorder.write_output_summary,
        TERecorder.write,
        Ticker.write,
        IntervalRecorder.write_genotypes,
        IntervalRecorder.write_phenotypes,
    )
    for specification in extract_output_specification_from_docstring(
        method=method
    )  # This loop is necessary in case there are multiple output specifications in one method
]
