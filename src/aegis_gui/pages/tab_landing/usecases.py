import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

# Content for each tab
tab_content = {
    "tab-1": "Content for Students:\n\nAEGIS is an excellent learning tool, offering students an opportunity to simulate life history traits and observe evolutionary processes in a controlled environment. Whether you're studying genetics, ecology, or evolutionary biology, AEGIS provides hands-on experience with real-world scenarios, enhancing theoretical knowledge with practical applications.",
    "tab-2": "Content for Theorists:\n\nTheorists can leverage AEGIS to test and refine hypotheses about life history evolution. The tool's flexibility allows for the exploration of various evolutionary scenarios, offering insights into the dynamics of selection, mutation, and environmental pressures. AEGIS serves as a bridge between theoretical models and empirical data.",
    "tab-3": "Content for Empirical Researchers:\n\nFor empirical researchers, AEGIS provides a powerful platform to simulate complex evolutionary processes. It allows for the testing of hypotheses before conducting real-world experiments, saving time and resources. The simulations can help in designing experiments and interpreting experimental data, making AEGIS an indispensable tool in evolutionary biology research.",
}

# Layout
layout = dbc.Card(
    [
        dbc.CardHeader(
            [
                dbc.Tabs(
                    [
                        dbc.Tab(label="Learning", tab_id="tab-1"),
                        dbc.Tab(label="Theoretical research", tab_id="tab-2"),
                        dbc.Tab(label="Applied research", tab_id="tab-3"),
                    ],
                    id="card-tabs",
                    active_tab="tab-1",
                ),
            ]
        ),
        dbc.CardBody(html.P(id="card-content", className="card-text")),
    ]
)


# Callback to update card content based on selected tab
@dash.callback(Output("card-content", "children"), Input("card-tabs", "active_tab"))
def update_card_content(active_tab):
    return tab_content.get(active_tab, "No content available.")
