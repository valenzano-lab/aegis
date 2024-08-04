import dash
import logging
import typing
import dash_bootstrap_components as dbc
from aegis_gui.utilities import log_funcs, utilities
from aegis_gui.pages.tab_config import config_input
from aegis.modules.initialization.parameterization.parameter import Parameter
from aegis.modules.initialization.parameterization.default_parameters import DEFAULT_PARAMETERS

# from aegis_gui.pages.tab_config.page import TEXTS_DOMAIN

from aegis.modules.recording.recordingmanager import RecordingManager
from aegis.modules.reproduction.reproduction import Reproducer
from aegis.modules.mortality import starvation, predation, infection, abiotic
from aegis.modules.genetics.composite.architecture import CompositeArchitecture
from aegis.modules.genetics.modifying.architecture import ModifyingArchitecture
from aegis.modules.genetics.envdrift import Envdrift

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


def get_pars():
    subsets = {domain: [] for domain in TEXTS_DOMAIN.keys()}
    for param in DEFAULT_PARAMETERS.values():
        if param.domain not in subsets:
            logging.error("Parameter domain {param.domain} has no documentation.")
            subsets[param.domain] = []
        subsets[param.domain].append(param)
    return subsets


def get_foldable():

    pars = get_pars()

    print(pars)

    elements = [
        dash.html.Div(
            [
                dbc.Button(
                    domain,
                    id={"type": "collapse-button", "index": domain},
                    className="mb-2",
                    color="light",
                    # outline=True,
                    n_clicks=0,
                ),
                dbc.Collapse(
                    dbc.Card(
                        dbc.CardBody(get_line(ps)),
                        # color="light",
                    ),
                    id={"type": "collapse-area", "index": domain},
                    is_open=False,
                ),
            ]
        )
        for domain, ps in pars.items()
    ]

    elements = [
        dbc.Accordion(
            [
                dbc.AccordionItem(
                    children=[
                        dbc.Modal(children=[TEXTS_DOMAIN.get(domain)]),
                        get_line(ps),
                    ],
                    id={"type": "collapse-button", "index": domain},
                    title=domain,
                )
                # dbc.AccordionItem(
                #     [
                #         dash.html.P("This is the content of the second section"),
                #         dbc.Button("Don't click me!", color="danger"),
                #     ],
                #     title="Item 2",
                # ),
                # dbc.AccordionItem(
                #     "This is the content of the third section",
                #     title="Item 3",
                # ),
                for domain, ps in pars.items()
            ]
        )
    ]

    return elements


@dash.callback(
    dash.Output({"type": "collapse-button", "index": dash.ALL}, "active"),
    dash.Output({"type": "collapse-area", "index": dash.ALL}, "is_open"),
    dash.Input({"type": "collapse-button", "index": dash.ALL}, "n_clicks"),
    dash.State({"type": "collapse-area", "index": dash.ALL}, "id"),
    prevent_initial_call=True,
)
def toggle_collapse(n, id_s):
    if n is None:
        return

    triggered_index = dash.ctx.triggered_id["index"]
    is_opens = [id_["index"] == triggered_index for id_ in id_s]

    return is_opens, is_opens
    # if n:
    #     return not is_open
    # return is_open


def get_line(ps: typing.List[Parameter]):
    elements = dash.html.Div(
        [
            dash.html.Div(
                [
                    dbc.Label(p.get_name()),
                    # p.key,
                    # p.dtype.__name__,
                    (
                        dbc.Button(
                            dash.html.I(className="bi bi-info-circle-fill"),
                            id={"type": "info-tooltip", "index": p.key},
                            color="link",
                        )
                        if p.info
                        else None
                    ),
                    (
                        dbc.Tooltip(
                            p.info,
                            target={"type": "info-tooltip", "index": p.key},
                            placement="right",
                        )
                        if p.info
                        else None
                    ),
                    config_input.get_input_element(param=p),
                ]
            )
            for p in ps
        ]
    )
    return elements
