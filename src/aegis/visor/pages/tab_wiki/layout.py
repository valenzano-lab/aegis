import dash
from dash import html, dcc

dash.register_page(__name__, path="/wiki", name="wiki")

layout = html.Div(
    children=[
        # TODO change text
        html.P(
            [
                """
                    This is the wiki tab.
                    """,
            ],
        )
    ],
)
