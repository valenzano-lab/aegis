import dash_bootstrap_components as dbc
from dash import html
from aegis.visor.pages.tab_plot.plot.prep_setup import FIG_SETUP

buttons = html.Div(
    [
        dbc.Button("Regular", color="primary", className="me-1"),
        dbc.Button("Active", color="primary", active=True, className="me-1"),
        dbc.Button("Disabled", color="primary", disabled=True),
    ]
)

gns = [graph_name for graph_name, d in FIG_SETUP.items() if d["supports_multi"] or not d["supports_multi"]]

select = dbc.Select(
    id="figure-select",
    options=[
        # {"label": "Option 1", "value": "1"},
        # {"label": "Option 2", "value": "2"},
        # {"label": "Disabled option", "value": "3", "disabled": True},
        {"label": gn, "value": gn}
        for gn in gns
    ],
    value=gns[3],
)
