"""
Functions for dynamic generation of documentation from source code.

"""

import pathlib
import pandas as pd
from aegis_sim.parameterization.default_parameters import DEFAULT_PARAMETERS
from aegis_gui.docs.specifications import output_specifications  # already dict_list
from aegis_gui.docs.domains import TEXTS_DOMAIN_MD

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


def dict_to_markdown(data_dict, output_file):
    with open(output_file, "w") as md_file:
        for key, value in data_dict.items():
            # Write the key as a markdown heading
            md_file.write(f"## {key.upper()}\n\n")

            # Write each item in the list as a paragraph
            md_file.write(f"{value}\n\n")  # Add double newline to separate paragraphs


write_dictlists_as_markdown_table(get_default_parameter_dictlists(), path=here / "default_parameters.md")
write_dictlists_as_markdown_table(output_specifications, path=here / "output_specifications.md")
dict_to_markdown(TEXTS_DOMAIN_MD, output_file=here / "submodel_specifications.md")
