import dash
import dash_bootstrap_components as dbc


def get_offcanvas_trigger():
    return dbc.Button(
        children=[dash.html.I(className="bi bi-gear-wide"), "Preferences"],
        id="open-offcanvas-backdrop",
        n_clicks=0,
        outline=True,
        color="primary",
        style={"marginTop": "2rem"},
    )


def get_theme_switch():
    return dash.html.Div(
        [
            dbc.Label("Dark mode", html_for="color-mode-switch"),
            dbc.Switch(id="color-mode-switch", value=False, className="d-inline-block ms-1", persistence=True),
        ]
    )


def get_delete_switch():
    return dash.html.Div(
        [
            dbc.Label("One-click deletion", html_for="one-click-deletion"),
            dbc.Switch(
                id="one-click-deletion",
                value=False,
                className="d-inline-block ms-1",
                persistence=True,
            ),
        ]
    )


def get_offcanvas():
    children = [
        get_theme_switch(),
        get_delete_switch(),
    ]
    return dbc.Offcanvas(
        children=children,
        id="offcanvas-backdrop",
        scrollable=True,
        title="User preferences",
    )


@dash.callback(
    dash.Output("offcanvas-backdrop", "is_open"),
    dash.Input("open-offcanvas-backdrop", "n_clicks"),
    dash.State("offcanvas-backdrop", "is_open"),
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open


dash.clientside_callback(
    """
    (switchOn) => {
       document.documentElement.setAttribute('data-bs-theme', switchOn ? 'dark' : 'false');  
       return window.dash_clientside.no_update
    }
    """,
    dash.Output("color-mode-switch", "id"),
    dash.Input("color-mode-switch", "value"),
)
