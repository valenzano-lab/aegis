import dash
import logging

from dash import html, dcc
from aegis_gui.utilities import log_funcs, utilities
from aegis_gui.pages.tab_config import config_input, run_simulation_button, use_prerun

from aegis_gui.pages.tab_config import foldable, copy_config

from aegis.modules.initialization.parameterization.default_parameters import DEFAULT_PARAMETERS
from aegis.modules.initialization.parameterization.parameter import Parameter

# Documentation sources
from aegis.modules.recording.recordingmanager import RecordingManager
from aegis.modules.reproduction.reproduction import Reproducer
from aegis.modules.mortality import starvation, predation, infection, abiotic
from aegis.modules.genetics.composite.architecture import CompositeArchitecture
from aegis.modules.genetics.modifying.architecture import ModifyingArchitecture
from aegis.modules.genetics.envdrift import Envdrift

dash.register_page(__name__, path="/config", name="config")

# TODO source from documentation
TEXTS_DOMAIN = {
    "starvation": utilities.extract_gui_from_docstring(starvation.Starvation),
    "predation": utilities.extract_gui_from_docstring(predation.Predation),
    "infection": utilities.extract_gui_from_docstring(infection.Infection),
    "abiotic": utilities.extract_gui_from_docstring(abiotic.Abiotic),
    "reproduction": utilities.extract_gui_from_docstring(Reproducer),
    "recording": utilities.extract_gui_from_docstring(RecordingManager),
    "genetics": """
    Every individual carries their own genome. In AEGIS, those are bit strings (arrays of 0's and 1's), passed on from parent to offspring, and mutated in the process.
    The submodel genetics transforms genomes into phenotypes; more specifically – into intrinsic phenotypes – biological potentials to exhibit a certain trait (e.g. probability to reproduce).
    These potentials are either realized or not realized, depending on the environment (e.g. availability of resources), interaction with other individuals (e.g. availability of mates) and interaction with other traits (e.g. survival).

    In AEGIS, genetics is simplified in comparison to the biological reality – it references no real genes and it simulates no molecular interactions; thus, it cannot be used to answer questions about specific genes, metabolic pathways or molecular mechanisms.
    However, in AEGIS, in comparison to empirical datasets, genes are fully functionally characterized (in terms of their impact on the phenotype), and are to be studied as functional, heritable genetic elements – in particular, their evolutionary dynamics.

    The configuration of genetics – the genetic architecture – is highly flexible. This includes specifying which traits are evolvable number of genetic elements (i.e. size of genome)...
    AEGIS offers two genetic architectures – composite and modifying. They are mutually exclusive and are described in detail below...
    """,
    "composite genetic architecture": utilities.extract_gui_from_docstring(CompositeArchitecture),
    "modifying genetic architecture": utilities.extract_gui_from_docstring(ModifyingArchitecture),
    "environmental drift": utilities.extract_gui_from_docstring(Envdrift),
    # "technical": "",
    # "other": "",
}


PREFACE = [
    html.Div(
        children=[
            # TODO change text
            html.H1("Configuration tab"),
            # html.P(
            #     [
            #         # """Customize and start simulations.
            #         # """
            #         # """
            #         # This is the configuration tab. Here you can customize parameters and start simulations.
            #         # On the left side, there are the parameter tables in which you can specify custom parameter values.
            #         # Each parameter has a valid data type and a valid range; when the input value is not valid, the simulation cannot be started.
            #         # If no input is given, the parameter will take on the default value.
            #         # Parameters are grouped into tables depending on which simulated process (submodel) they customize – these are, in brief, mortality (infection, predation, starvation, abiotic mortality), reproduction, genetics, environmental drift and recording.
            #         # Each submodel is described on the right side, next to the relevant parameter table.
            #         # To start the simulation, adjust parameter values, enter a unique ID in the input bar on top and click the adjacent button "run simulation".
            #         # """,
            #     ],
            #     style={"margin-bottom": "2rem"},
            # ),
        ]
    )
]

HEADER = html.Tr(
    [
        html.Th("PARAMETER", style={"padding-left": "1.2rem"}),
        html.Th("TYPE"),
        html.Th("RANGE", className="valid-values"),
        html.Th("VALUE"),
        # html.Th("DESCRIPTION", style={"padding-right": "1.2rem"}),
    ],
)


def layout() -> html.Div:
    # Group parameters by domain
    subsets = {domain: [] for domain in TEXTS_DOMAIN.keys()}
    for param in DEFAULT_PARAMETERS.values():
        if param.domain not in subsets:
            logging.error("Parameter domain {param.domain} has no documentation.")
            subsets[param.domain] = []
        subsets[param.domain].append(param)

    # Generate layout

    # tables = []
    # for domain, subset in subsets.items():
    #     tables.append(
    #         html.Div(
    #             children=[
    #                 html.P(domain, className="config-domain"),
    #                 html.Div(
    #                     children=[
    #                         html.Div(
    #                             children=[
    #                                 html.Div(
    #                                     children=[get_table(subset)],
    #                                     # style={"width": "50%"},
    #                                 ),
    #                             ],
    #                         ),
    #                         html.Div(
    #                             html.Div(children=TEXTS_DOMAIN.get(domain, ""), className="config-domain-desc"),
    #                             style={"margin-left": "1.5rem"},
    #                         ),
    #                     ],
    #                     style={"display": "flex"},
    #                 ),
    #             ],
    #         )
    #     )

    foldables = foldable.get_foldable()

    # Generate layout
    return html.Div(
        id="sim-section",
        # style={"display": "none"},
        children=PREFACE
        + copy_config.make_select()
        + foldables
        + [use_prerun.make_select()]
        + [run_simulation_button.layout],
    )


# def get_row(v: Parameter) -> html.Tr:
#     if v.serverrange_info:
#         serverrange_info_message = f"Allowed parameter range for the server is {v.serverrange_info}."
#     else:
#         serverrange_info_message = ""

#     return html.Tr(
#         [
#             # PARAMETER
#             html.Td(
#                 v.get_name(),
#                 style={"padding-left": "1.2rem"},
#                 title=v.info if v.info else None,
#             ),
#             # TYPE
#             html.Td(
#                 children=html.Label(
#                     v.dtype.__name__,
#                     className=f"dtype-{v.dtype.__name__} dtype",
#                 )
#             ),
#             # RANGE
#             html.Td(children=v.drange, className="data-range"),
#             # VALUE
#             html.Td(children=config_input.get_input_element(param=v)),
#             # DESCRIPTION
#             # html.Td(
#             #     children=[
#             #         v.info + ".",
#             #         html.P(
#             #             serverrange_info_message,
#             #             className="serverrange_info_message",
#             #         ),
#             #     ],
#             #     className="td-info",
#             #     style={"padding-right": "0.8rem"},
#             # ),
#         ],
#     )


# def get_table(params_subset) -> html.Table:
#     return html.Table(
#         className="config-table",
#         children=[
#             html.Col(className="config-table-col"),
#             html.Col(className="config-table-col"),
#             html.Col(className="config-table-col"),
#             html.Col(className="config-table-col"),
#             html.Col(className="config-table-col"),
#         ]
#         + [HEADER]
#         + [get_row(v) for v in params_subset if not isinstance(v.default, list)],
#     )
