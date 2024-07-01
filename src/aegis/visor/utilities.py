import platformdirs
import subprocess

import pathlib
import yaml
import logging
import re

import pandas as pd

from dash import html, dcc

from aegis.modules.initialization.parameterization.default_parameters import DEFAULT_PARAMETERS

# TODO ensure that there is default dataset available
default_selection_states = (["default", True],)


def get_here():
    return pathlib.Path(__file__).absolute().parent


def get_base_dir():
    return pathlib.Path(platformdirs.user_data_dir("aegis", "aegis"))


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


def run(filename):
    config_path = get_config_path(filename)
    logging.info(f"Running a simulation at path {config_path}.")
    subprocess.Popen(["python3", "-m", "aegis", "--config_path", config_path])


def make_config_file(filename, configs):
    configs["PHENOMAP_SPECS"] = []
    configs["NOTES"] = []
    for k, v in configs.items():
        configs[k] = DEFAULT_PARAMETERS[k].convert(v)
    logging.info("Making a config file.")
    config_path = get_config_path(filename)
    with open(config_path, "w") as file_:
        yaml.dump(configs, file_)


def log_debug(func):
    def wrapper(*args, **kwargs):
        logging.debug(f"Executing function: {func.__name__}.")
        return func(*args, **kwargs)

    return wrapper


def log_info(func):
    def wrapper(*args, **kwargs):
        logging.info(f"Executing function: {func.__name__}.")
        return func(*args, **kwargs)

    return wrapper


def get_sim_paths():
    sim_dir = get_sim_dir()
    return [p for p in sim_dir.iterdir() if p.is_dir()]


def get_sims():
    return [p.stem for p in get_sim_paths()]


def sim_exists(filename: str) -> bool:
    paths = get_sim_paths()
    return any(path.stem == filename for path in paths)


def extract_visor_from_docstring(class_):
    docstring = class_.__doc__
    visor_section = docstring.split("VISOR")[1]
    # text = replace_params_in_brackets_with_span(visor_section)
    parsed = parse_visor_docstrings(visor_section)
    return list(parsed)


def extract_output_specification_from_docstring(method):
    """Extract information about the output file created by the method"""
    docstring = method.__doc__
    text = docstring.split("# OUTPUT SPECIFICATION")[1]
    parsed = {}
    for pair in text.strip().split("\n"):
        k, v = pair.split(":", maxsplit=1)
        parsed[k.strip()] = v.strip()
    return parsed


# def return_output_specifications_as_markdown_table(methods):


def parse_visor_docstrings(text):
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
from aegis.modules.recording.visorrecorder import VisorRecorder

OUTPUT_SPECIFICATIONS = [
    extract_output_specification_from_docstring(method=method)
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
        VisorRecorder.write_genotypes,
        VisorRecorder.write_phenotypes,
    )
]
