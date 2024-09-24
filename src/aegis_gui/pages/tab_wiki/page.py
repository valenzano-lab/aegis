import dash
from dash import html, dash_table
from aegis_gui.docs.specifications import output_specifications
import dash_bootstrap_components as dbc

from aegis_sim.submodels import genetics
import pathlib
import aegis


dash.register_page(__name__, path="/wiki", name="AEGIS | Wiki")


def make_output_specification_table():
    """

    Documentation for plotly datatables: https://dash.plotly.com/datatable
    """
    data = [
        {key: specs.get(key, "!!! nan") for key in output_specifications[0].keys()} for specs in output_specifications
    ]
    columns = [{"id": c, "name": c} for c in output_specifications[0].keys()]
    return dash_table.DataTable(
        data=data,
        columns=columns,
        style_data={"whiteSpace": "normal", "height": "auto"},
    )


def layout() -> html.Div:
    return html.Div(
        [
            html.H1("""Wiki tab"""),
            dbc.Select(
                id="select",
                options=[
                    {"label": "Specification of the genetic architecture in AEGIS", "value": "genarch"},
                    {"label": "Specification of output files", "value": "output"},
                    {"label": "Specification of input parameters", "value": "input"},
                    {"label": "Specification of submodels", "value": "submodels"},
                    # {"label": "Disabled option", "value": "3", "disabled": True},
                ],
                value="input",
                style={"marginBottom": "1rem"},
            ),
            html.Div(id="select-container", children=[]),
        ]
    )


@dash.callback(
    dash.Output("select-container", "children"),  # Updates the content of the container div
    [dash.Input("select", "value")],  # Listens to changes in the Select dropdown
)
def update_select_container(selected_value):
    if selected_value == "output":
        return [
            # html.H6("Specification of output files"),
            make_accordion_specs_output_files(),  # Call the appropriate function to return content
        ]
    elif selected_value == "genarch":
        return [
            # html.H6("Specification of the genetic architecture in AEGIS"),
            dash.dcc.Markdown(
                children=get_specification_of_gen_arch(), mathjax=True, dangerously_allow_html=True
            ),  # Display Markdown with MathJax
        ]
    elif selected_value == "input":
        path_to_dynamic = pathlib.Path(aegis.__file__).parent / "documentation" / "dynamic"
        with open(path_to_dynamic / "default_parameters.md", "r") as file_:
            text = file_.read()
        return dash.dcc.Markdown(children=text)
    elif selected_value == "submodels":
        path_to_dynamic = pathlib.Path(aegis.__file__).parent / "documentation" / "dynamic"
        with open(path_to_dynamic / "submodel_specifications.md", "r") as file_:
            text = file_.read()
        return dash.dcc.Markdown(children=text)
    else:
        return html.P("Please select an option.")  # Default message for no selection or invalid option


def make_accordion_item(d):

    return dbc.ListGroup(
        children=[dbc.ListGroupItem(children=[dash.html.Strong(f"{k}: "), v]) for k, v in d.items() if k != "path"],
        flush=True,
    )


def make_accordion_specs_output_files():
    data = [
        {key: specs.get(key, "!!! nan") for key in output_specifications[0].keys()} for specs in output_specifications
    ]
    return dbc.Accordion(
        id="wiki-accordion",
        children=[dbc.AccordionItem(title=d["path"], children=make_accordion_item(d)) for d in data],
        start_collapsed=True,
    )


def get_specification_of_gen_arch():
    path_to_md = pathlib.Path(genetics.__file__).parent / "doc.md"
    with open(path_to_md, "r") as file_:
        return file_.read()
