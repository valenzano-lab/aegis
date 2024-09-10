"""
Functions for dynamic generation of documentation from source code.

"""

import pathlib
import pandas as pd
from aegis_sim.modules.initialization.parameterization.default_parameters import DEFAULT_PARAMETERS
from aegis_gui.utilities.utilities import OUTPUT_SPECIFICATIONS  # already dict_list

here = pathlib.Path(__file__).parent


def get_default_parameter_dictlists():
    cols = ["key", "name", "default", "info", "dtype", "drange"]
    dictlists = [{col: getattr(p, col) for col in cols} for p in DEFAULT_PARAMETERS.values()]
    return dictlists


def write_dictlists_as_markdown_table(dict_list, path):
    # Requires tabulate package
    # dict_list like OUTPUT_SPECIFICATIONS; [{col1: val1, col2, val2, ...}, {col1: val1', col2: val2', ...}, ...]
    df = pd.DataFrame(dict_list)
    md = df.to_markdown()
    with open(path, "w") as file_:
        file_.writelines(md)


write_dictlists_as_markdown_table(get_default_parameter_dictlists(), path=here / "default_parameters.md")
write_dictlists_as_markdown_table(OUTPUT_SPECIFICATIONS, path=here / "output_specifications.md")
