import subprocess
import sys

import pathlib
import yaml
import logging
import re

from dash import html

from aegis.utilities.container import Container
from aegis.modules.initialization.parameterization.default_parameters import DEFAULT_PARAMETERS

# TODO ensure that there is default dataset available

running_processes = {}


def get_here():
    return pathlib.Path(__file__).absolute().parent


def get_base_dir():
    base_dir = (pathlib.Path.home() / "aegis_data").absolute()
    base_dir.mkdir(exist_ok=True, parents=True)
    # base_dir = pathlib.Path(platformdirs.user_data_dir("aegis", "aegis"))
    # base_dir.mkdir(exist_ok=True, parents=True)
    return base_dir


def get_sim_dir():
    base_dir = get_base_dir() / "sim_data"
    base_dir.mkdir(exist_ok=True)
    return base_dir


def get_figure_dir():
    figure_dir = get_base_dir() / "figures"
    figure_dir.mkdir(exist_ok=True)
    return figure_dir


def read_yml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_config_path(filename):
    return get_sim_dir() / f"{filename}.yml"


def run_simulation(filename, prerun_sim_path):
    global running_processes
    config_path = get_config_path(filename)
    logging.info(f"Running a simulation at path {config_path}.")
    if prerun_sim_path is None:
        pickle_command = []
    else:
        container = Container(prerun_sim_path)
        latest_pickle_path = container.paths["pickles"][-1]
        logging.info(f"Using pickled population from {latest_pickle_path}.")
        pickle_command = ["-p", latest_pickle_path]
    process = subprocess.Popen([sys.executable, "-m", "aegis", "--config_path", config_path] + pickle_command)
    running_processes[filename] = process


def terminate_simulation(simname):
    container = get_container(filename=simname)
    container.terminate()


def get_container(filename):
    return Container(get_sim_dir() / filename)


def make_config_file(filename, configs):
    configs["PHENOMAP_SPECS"] = []
    configs["NOTES"] = []
    for k, v in configs.items():
        configs[k] = DEFAULT_PARAMETERS[k].convert(v)
    logging.info("Making a config file.")
    config_path = get_config_path(filename)
    with open(config_path, "w") as file_:
        yaml.dump(configs, file_)


def get_sim_paths():
    sim_dir = get_sim_dir()
    paths = [p for p in sim_dir.iterdir() if p.is_dir()]
    paths = sorted(paths, key=lambda path: path.name)
    return paths


def get_sims():
    return [p.stem for p in get_sim_paths()]


def sim_exists(filename: str) -> bool:
    paths = get_sim_paths()
    return any(path.stem == filename for path in paths)


def extract_gui_from_docstring(class_):
    docstring = class_.__doc__
    gui_section = docstring.split("GUI")[1]
    # text = replace_params_in_brackets_with_span(gui_section)
    parsed = parse_gui_docstrings(gui_section)
    return list(parsed)


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


# def return_output_specifications_as_markdown_table(methods):


def parse_gui_docstrings(text):
    pattern = r"(\[\[|\]\])"
    parts = re.split(pattern, text)
    is_parameter = False
    for part in parts:
        if part == "[[":
            is_parameter = True
        elif part == "]]":
            is_parameter = False
        else:
            if is_parameter:
                yield get_parameter_span(part)
            else:
                yield html.Span(part)


def get_parameter_span(name):
    reformatted_name = name.replace("_", " ").lower()
    param = DEFAULT_PARAMETERS[name]
    info = param.info
    return html.Span(
        reformatted_name,
        title=info,
        # style={"font-style": "italic"},
    )


# Output specification sources
from aegis.modules.recording.featherrecorder import FeatherRecorder
from aegis.modules.recording.flushrecorder import FlushRecorder
from aegis.modules.recording.phenomaprecorder import PhenomapRecorder
from aegis.modules.recording.picklerecorder import PickleRecorder
from aegis.modules.recording.popgenstatsrecorder import PopgenStatsRecorder
from aegis.modules.recording.progressrecorder import ProgressRecorder
from aegis.modules.recording.summaryrecorder import SummaryRecorder
from aegis.modules.recording.terecorder import TERecorder
from aegis.modules.recording.ticker import Ticker
from aegis.modules.recording.intervalrecorder import IntervalRecorder

OUTPUT_SPECIFICATIONS = [
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
