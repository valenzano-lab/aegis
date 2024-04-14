from dash import html, dcc
from visor import utilities
from aegis.modules.initialization.parameterization.default_parameters import DEFAULT_PARAMETERS
from aegis.modules.initialization.parameterization.parameter import Parameter

from aegis.modules.recording.recordingmanager import RecordingManager
from aegis.modules.reproduction.reproduction import Reproducer
from aegis.modules.mortality import starvation, predation, infection, abiotic
from aegis.modules.genetics.composite.architecture import CompositeArchitecture
from aegis.modules.genetics.modifying.architecture import ModifyingArchitecture
from aegis.modules.genetics.envdrift import Envdrift

# TODO source from documentation
TEXTS_DOMAIN = {
    "infection": utilities.extract_visor_from_docstring(infection.Infection),
    "predation": utilities.extract_visor_from_docstring(predation.Predation),
    "reproduction": utilities.extract_visor_from_docstring(Reproducer),
    "starvation": utilities.extract_visor_from_docstring(starvation.Starvation),
    "abiotic": utilities.extract_visor_from_docstring(abiotic.Abiotic),
    "recording": utilities.extract_visor_from_docstring(RecordingManager),
    "genetics": "",
    "composite genetic architecture": utilities.extract_visor_from_docstring(CompositeArchitecture),
    "modifying genetic architecture": utilities.extract_visor_from_docstring(ModifyingArchitecture),
    "environmental drift": utilities.extract_visor_from_docstring(Envdrift),
    "other": "",
}


PREFACE = [
    html.Div(
        children=[
            # TODO change text
            html.P(
                [
                    """
        Using this tab you can customize your simulation and run it.
        Change the parameter values under the column name VALUE.
        Run the simulation by giving it a unique id name and clicking the button 'run simulation'.
        """
                ]
            )
        ],
    )
]

HEADER = html.Tr(
    [
        html.Th("PARAMETER", style={"padding-left": "1.2rem"}),
        html.Th("VALUE"),
        html.Th("TYPE"),
        html.Th("RANGE", className="valid-values"),
        html.Th("DESCRIPTION", style={"padding-right": "1.2rem"}),
    ],
)


@utilities.log_debug
def get_config_layout() -> html.Div:
    # Group parameters by domain
    subsets = {domain: [] for domain in TEXTS_DOMAIN.keys()}
    for param in DEFAULT_PARAMETERS.values():
        subsets[param.domain].append(param)

    # Generate layout

    tables = []
    for domain, subset in subsets.items():
        tables.extend(
            [
                html.Div(
                    children=[
                        html.P(domain, className="config-domain"),
                        html.P(TEXTS_DOMAIN[domain], className="config-domain-desc"),
                    ],
                ),
                get_table(subset),
            ]
        )

    # Generate layout
    return html.Div(
        id="sim-section",
        style={"display": "none"},
        children=PREFACE + tables,
    )


# @utilities.log_debug
def get_row(v: Parameter) -> html.Tr:
    if v.serverrange_info:
        serverrange_info_message = f"Allowed parameter range for the server is {v.serverrange_info}."
    else:
        serverrange_info_message = ""

    return html.Tr(
        [
            # PARAMETER
            html.Td(v.get_name(), style={"padding-left": "1.2rem"}),
            # VALUE
            html.Td(
                children=dcc.Input(
                    type="text",
                    placeholder=str(v.default) if v.default is not None else "",
                    # id=f"config-{v.key}",
                    id={"type": "config-input", "index": v.key},
                    autoComplete="off",
                    className="config-input-class",
                ),
            ),
            # TYPE
            html.Td(
                children=html.Label(
                    v.dtype.__name__,
                    className=f"dtype-{v.dtype.__name__} dtype",
                )
            ),
            # RANGE
            html.Td(children=v.drange, className="data-range"),
            # DESCRIPTION
            html.Td(
                children=[
                    v.info + ".",
                    html.P(
                        serverrange_info_message,
                        className="serverrange_info_message",
                    ),
                ],
                className="td-info",
                style={"padding-right": "0.8rem"},
            ),
        ],
    )


@utilities.log_debug
def get_table(params_subset) -> html.Table:
    return html.Table(
        className="config-table",
        children=[
            html.Col(className="config-table-col"),
            html.Col(className="config-table-col"),
            html.Col(className="config-table-col"),
            html.Col(className="config-table-col"),
            html.Col(className="config-table-col"),
        ]
        + [HEADER]
        + [get_row(v) for v in params_subset if not isinstance(v.default, list)],
    )
