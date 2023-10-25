from dash import html, dcc
from aegis.help import config

# TODO change text
texts_domain = {
    "recording": "Change which data are recorded and with what frequency.",
    "predation": "asdf",
    "computation": "wer",
    "genetics": "asdf",
    "initialization": "wer",
    "infection": "asdf",
    "ecology": "wer",
    "environment": "asdf",
}

assert set(config.get_domains()) == set(
    texts_domain
), "Specified and expected domains do not match"

preface = [
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
        style={"color": "white"},
    )
]

header = html.Tr(
    [
        html.Th("PARAMETER", style={"padding-left": "1.2rem"}),
        html.Th("VALUE"),
        html.Th("TYPE"),
        html.Th("RANGE", className="valid-values"),
        # html.Th("DOMAIN"),
        html.Th("DESCRIPTION", style={"padding-right": "1.2rem"}),
    ],
)


def get_config_section():
    # Group parameters by domain
    subsets = {domain: [] for domain in texts_domain.keys()}
    for param in config.params.values():
        subsets[param.domain].append(param)

    # Generate layout

    tables = []
    for domain, subset in subsets.items():
        tables.extend(
            [
                html.Div(
                    children=[
                        html.P(domain, className="config-domain"),
                        html.P(texts_domain[domain], className="config-domain-desc"),
                    ],
                ),
                get_table(subset),
            ]
        )

    # Generate layout
    return html.Div(
        id="sim-section",
        children=preface + tables,
    )


def get_row(v):
    return html.Tr(
        [
            html.Td(v.get_name(), style={"padding-left": "1.2rem"}),
            html.Td(
                children=dcc.Input(
                    type="text",
                    placeholder=str(v.default) if v.default is not None else "",
                    id=f"config-{v.key}",
                    autoComplete="off",
                ),
            ),
            html.Td(
                children=html.Label(
                    v.dtype.__name__,
                    className=f"dtype-{v.dtype.__name__} dtype",
                )
            ),
            html.Td(children=v.drange, className="data-range"),
            # html.Td(
            #     children=html.Label(
            #         v.domain,
            #         className=f"domain-{v.domain} domain",
            #     ),
            # ),
            html.Td(
                v.info,
                className="td-info",
                style={"padding-right": "0.8rem"},
            ),
        ],
    )


def get_table(params_subset):
    return html.Table(
        className="config-table",
        children=[header]
        + [get_row(v) for v in params_subset if not isinstance(v.default, list)],
    )
