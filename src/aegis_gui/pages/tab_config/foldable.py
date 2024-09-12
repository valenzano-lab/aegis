import dash
import logging
import typing
import dash_bootstrap_components as dbc
from aegis_gui.pages.tab_config import config_input
from aegis_sim.parameterization.parameter import Parameter
from aegis_sim.parameterization.default_parameters import DEFAULT_PARAMETERS

from aegis_gui.docs.domains import TEXTS_DOMAIN


def get_pars():
    subsets = {domain: [] for domain in TEXTS_DOMAIN.keys()}
    for param in DEFAULT_PARAMETERS.values():
        if param.domain not in subsets:
            logging.error(f"Parameter domain {param.domain} has no documentation.")
            subsets[param.domain] = []
        subsets[param.domain].append(param)
    return subsets


def get_foldable():

    pars = get_pars()

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
                        dbc.CardBody(get_line(ps, domain)),
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
        dash.html.H6("Customize parameters"),
        dbc.Accordion(
            id="config-accordion",
            start_collapsed=True,
            children=[
                dbc.AccordionItem(
                    children=[
                        dbc.Modal(
                            children=[
                                dbc.ModalHeader(dbc.ModalTitle(f"{domain.title()} module")),
                                dbc.ModalBody(TEXTS_DOMAIN.get(domain)),
                                # dbc.ModalFooter(),
                            ],
                            id={"type": "domain-info-modal", "index": domain},
                            className="domain-modal-info",
                        ),
                        dash.html.Div(
                            [
                                "more information at ",
                                dbc.Button(
                                    dash.html.I(className="bi bi-info-square-fill"),
                                    id={"type": "domain-info-modal-trigger", "index": domain},
                                    color="link",
                                    style={"margin": 0, "padding": "0 0 0 0px"},
                                ),
                            ]
                        ),
                        get_line(ps, domain),
                    ],
                    id={"type": "collapse-button", "index": domain},
                    title=dash.html.Div(
                        [
                            domain.title(),
                            dash.html.Span(" "),
                            dbc.Badge(
                                "0 modifications",
                                pill=True,
                                className="ms-1",
                                color="secondary",
                                id={"type": "collapse-badge", "index": domain},
                            ),
                        ],
                        className="position-relative",
                    ),
                )
                for domain, ps in pars.items()
            ],
            # flush=True,
        ),
    ]

    return elements


@dash.callback(
    dash.Output({"type": "domain-info-modal", "index": dash.MATCH}, "is_open"),
    dash.Input({"type": "domain-info-modal-trigger", "index": dash.MATCH}, "n_clicks"),
    prevent_initial_call=True,
)
def toggle_modal(_):
    return True


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


def get_line(ps: typing.List[Parameter], domain):
    elements = dash.html.Div(
        [
            dash.html.Div(
                [
                    # dbc.Label(p.get_name()),
                    # p.key,
                    # p.dtype.__name__,
                    # (
                    #     dbc.Button(
                    #         dash.html.I(className="bi bi-info-square-fill"),
                    #         id={"type": "info-tooltip", "index": p.key},
                    #         color="link",
                    #     )
                    #     if p.info
                    #     else None
                    # ),
                    # (
                    #     dbc.Tooltip(
                    #         p.info,
                    #         target={"type": "info-tooltip", "index": p.key},
                    #         placement="right",
                    #     )
                    #     if p.info
                    #     else None
                    # ),
                    config_input.get_input_element(param=p),
                ]
            )
            for p in ps
        ],
    )
    return elements
