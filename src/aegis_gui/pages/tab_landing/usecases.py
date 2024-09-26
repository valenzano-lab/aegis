import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

tab_content = {
    "tab-educational": dash.html.Div(
        [
            dash.html.P(),
            dash.html.Ul(
                [
                    dash.html.P(
                        "AEGIS offers easy-access exploration of topics pertaining to evolution of aging and life history, as well as broader evolutionary topics such as genetic drift, predator-prey dynamics and population dynamics."
                    ),
                    # dash.html.Li(
                    #     "For educators and students, AEGIS is a unique platform for teaching and learning a wide range of concepts in a more intuitive manner, reducing the reliance on advanced mathematical or niche knowledge."
                    # ),
                    # dash.html.Li(
                    #     "Individual-based models, such as those demonstrating Lotka-Volterra predator-prey dynamics, are already established in education, and simulations of genetic drift or epidemiological models are common."
                    # ),
                    # dash.html.Li(
                    #     "This approach focuses on modeling relevant entities and processes, keeping technical and operational details secondary."
                    # ),
                    # dash.html.Li(
                    #     "AEGIS lowers the barrier to entry by offering a web-based interface or simple installation on personal computers, complete with built-in analytics and familiar visualizations."
                    # ),
                ]
            ),
        ]
    ),
    "tab-theoretical": dash.html.Div(
        [
            dash.html.Ul(
                [
                    dash.html.Li(
                        "Address complex and challenging topics that are difficult to model with traditional, analytical methods."
                    ),
                    dash.html.Li(
                        "Explore intricate genetic interactions, fluctuating environmental conditions, and population and evolutionary dynamics."
                    ),
                    dash.html.Li("Test hypotheses and generalize established theories under relaxed assumptions."),
                    dash.html.Li(
                        "Provide new perspectives on the evolution of accelerated aging and longevity, and study customizable genetic architectures of aging, with or without pleiotropy, covering genetic variants as described in mutation accumulation theory, antagonistic pleiotropy theory, and disposable soma theory."
                    ),
                    dash.html.Li(
                        "Generate simulated datasets for testing and refinement of analytical and computational tools for evolutionary biology."
                    ),
                ]
            )
        ]
    ),
    "tab-applied": dash.html.Div(
        [
            dash.html.Ul(
                [
                    dash.html.P(
                        "As a complementary method that is comparatively inexpensive, fast, and does not require animals, AEGIS offers independent study of empirical findings. It can be used before or after running experiments."
                    ),
                    dash.html.P("Before running experiments:", className="fw-bold mb-1"),
                    dash.html.Li(
                        "Assist in formulating hypotheses (models to-be-tested) by helping think through an evolutionary problem and identify relevant factors."
                    ),
                    dash.html.Li(
                        "Explore interactions among relevant evolutionary factors, gauge their relative impact, and test the plausibility of different hypotheses."
                    ),
                    dash.html.Li("Help estimate relevant evolutionary time scales and effect sizes."),
                    dash.html.P("After running experiments:", className="mt-3 fw-bold mb-1"),
                    dash.html.Li(
                        "Study generality and broader relevance of conducted experiments by simulating scenarios with comparable, loosened, or altered assumptions and comparing outcomes with observations."
                    ),
                ]
            )
        ]
    ),
}

# tab_content = {
#     "tab-1": "Content for Students:\n\nAEGIS is an excellent learning tool, offering students an opportunity to simulate life history traits and observe evolutionary processes in a controlled environment. Whether you're studying genetics, ecology, or evolutionary biology, AEGIS provides hands-on experience with real-world scenarios, enhancing theoretical knowledge with practical applications.",
#     "tab-2": "Content for Theorists:\n\nTheorists can leverage AEGIS to test and refine hypotheses about life history evolution. The tool's flexibility allows for the exploration of various evolutionary scenarios, offering insights into the dynamics of selection, mutation, and environmental pressures. AEGIS serves as a bridge between theoretical models and empirical data.",
#     "tab-3": "Content for Empirical Researchers:\n\nFor empirical researchers, AEGIS provides a powerful platform to simulate complex evolutionary processes. It allows for the testing of hypotheses before conducting real-world experiments, saving time and resources. The simulations can help in designing experiments and interpreting experimental data, making AEGIS an indispensable tool in evolutionary biology research.",
# }


# path_here = pathlib.Path(__file__)
# path_md = path_here.parents[3] / "aegis" / "documentation" / "1 who is aegis for.md"

# with open(path_md, "r") as file_:
#     text = markdown.markdown(file_.read())

# Layout
layout = dbc.Card(
    [
        dbc.CardHeader(
            [
                dbc.Tabs(
                    [
                        dbc.Tab(label="Theoretical research", tab_id="tab-theoretical"),
                        dbc.Tab(label="Applied research", tab_id="tab-applied"),
                        dbc.Tab(label="Learning", tab_id="tab-educational"),
                    ],
                    id="card-tabs",
                    active_tab="tab-theoretical",
                    persistence=True,
                ),
            ]
        ),
        dbc.CardBody(dash.html.P(id="card-content", className="card-text")),
    ]
)


# Callback to update card content based on selected tab
@dash.callback(Output("card-content", "children"), Input("card-tabs", "active_tab"))
def update_card_content(active_tab):
    return tab_content[active_tab]
