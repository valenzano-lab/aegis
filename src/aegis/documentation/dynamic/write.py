"""
Functions for dynamic generation of documentation from source code.

"""

import pathlib
import pandas as pd
from aegis_sim.parameterization.default_parameters import DEFAULT_PARAMETERS
from aegis_gui.docs import specifications
from aegis_gui.docs.domains import TEXTS_DOMAIN_MD

here = pathlib.Path(__file__).parent


def get_default_parameter_dictlists():
    cols = [
        "key",
        # "name",
        "default",
        "info",
        "dtype",
        "drange",
        "domain",
    ]
    dictlists = [{col: getattr(p, col) for col in cols} for p in DEFAULT_PARAMETERS.values() if p.show_in_docs]
    return dictlists


parameter_specifications_preamble = """
# Parameter specification

There are three kinds of properties in AEGIS: fixed, parametrizable and variable.
- Fixed properties, such as lack of mating preference, is hard-coded in the current version of AEGIS – it is valid for all simulations and all individuals and cannot be changed.
- Parametrizable properties, such as reproduction mode, is parametrizable – it can be set by a parameter (`REPRODUCTION_MODE`) to various values including `sexual` and `asexual`.
- All other properties, e.g. fertility (biological) or abiotic mortality (environmental), are variable – they differ between individuals and they vary over simulation time.  

The table below provides information about all parameters in AEGIS. Here we provide an explanation of table columns.
- `key` specifies the parameter names as they are given in the GUI and as they are expected in the .yml input configuration files.
- `default` specifies the default value of the parameter. This is the value that will be used if the user does not provide any value.
- `info` specifies a short description of the parameter. Note that to understand some of the parameters it is necessary to understand how they modify the behavior of the submodel they affect.
- `dtype` specifies the data type of the parameter value.
- `drange` specifies the valid range of values that can be used for the parameter.
- `domain` specifies the simulation component which the parameter modifies.

"""


def write_dictlists_as_markdown_table(dict_list, path, preamble):
    # Requires tabulate package
    # dict_list like OUTPUT_SPECIFICATIONS; [{col1: val1, col2, val2, ...}, {col1: val1', col2: val2', ...}, ...]
    df = pd.DataFrame(dict_list)
    md = df.to_markdown()
    with open(path, "w") as file_:
        if preamble:
            file_.writelines(preamble)
        file_.writelines(md)


def dict_to_markdown(data_dict, output_file):
    with open(output_file, "w") as md_file:
        for key, value in data_dict.items():
            # Write the key as a markdown heading
            md_file.write(f"## {key.upper()}\n\n")

            # Write each item in the list as a paragraph
            md_file.write(f"{value}\n\n")  # Add double newline to separate paragraphs


write_dictlists_as_markdown_table(
    get_default_parameter_dictlists(),
    path=here / "default_parameters.md",
    preamble=parameter_specifications_preamble,
)
write_dictlists_as_markdown_table(
    specifications.output_specifications,
    path=here / "output_specifications.md",
    preamble=specifications.output_specifications_preamble,
)
dict_to_markdown(
    TEXTS_DOMAIN_MD,
    output_file=here / "submodel_specifications.md",
)
