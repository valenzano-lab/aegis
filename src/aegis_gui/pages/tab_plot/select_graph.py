import dash_bootstrap_components as dbc
from dash import html
from aegis_gui.pages.tab_plot.plot.prep_setup import FIG_SETUP

buttons = html.Div(
    [
        dbc.Button("Regular", color="primary", className="me-1"),
        dbc.Button("Active", color="primary", active=True, className="me-1"),
        dbc.Button("Disabled", color="primary", disabled=True),
    ]
)

gns = [graph_name for graph_name, d in FIG_SETUP.items() if d["supports_multi"] or not d["supports_multi"]]

initial_graph = gns[3]

select = dbc.InputGroup(
    [
        dbc.InputGroupText("Plotting function"),
        dbc.Select(
            id="figure-select",
            options=[{"label": gn, "value": gn} for gn in gns],
            value=initial_graph,
            className="plot-dropdown",
        ),
    ]
)
