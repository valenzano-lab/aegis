import dash
import dash_bootstrap_components as dbc


inputgroup = dbc.InputGroup(
    dbc.Button(
        [dash.html.I(className="bi bi-arrow-clockwise"), "Refresh"],
        id="refresh-figure-data",
    )
)
