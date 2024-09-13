import dash
import dash_bootstrap_components as dbc

left_jumbotron = dbc.Col(
    dash.html.Div(
        [
            dash.html.H2("Simulate", className="display-3"),
            dash.html.Hr(className="my-2"),
            dash.html.P("Jump right in and run evolutionary simulations of life history."),
            dash.dcc.Link(dbc.Button("Start", color="light", outline=True), href="/config"),
        ],
        className="h-100 p-5 text-white bg-primary rounded-3",
    ),
    md=6,
)

right_jumbotron = dbc.Col(
    dash.html.Div(
        [
            dash.html.H2("Analyze", className="display-3"),
            dash.html.Hr(className="my-2"),
            dash.html.P("Or, inspect results of pre-run simulations."),
            dash.dcc.Link(
                dbc.Button(
                    "Show me plots",
                    color="light",
                    outline=True,
                    id="read-jumbotron-button",
                ),
                href="/plot",
            ),
        ],
        className="h-100 p-5 bg-secondary text-white border rounded-3",
    ),
    md=6,
)

jumbotron = dbc.Row(
    [left_jumbotron, right_jumbotron],
    className="align-items-md-stretch",
)
