import dash
from dash import html, dash_table
from aegis_gui.docs.specifications import output_specifications
import dash_bootstrap_components as dbc

from aegis_sim.submodels import genetics
import pathlib


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
                    {"label": "Specification of output files", "value": "1"},
                    {"label": "Specification of the genetic architecture in AEGIS", "value": "2"},
                    # {"label": "Disabled option", "value": "3", "disabled": True},
                ],
                value="1",
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
    if selected_value == "1":
        return [
            # html.H6("Specification of output files"),
            make_accordion_specs_output_files(),  # Call the appropriate function to return content
        ]
    elif selected_value == "2":
        return [
            # html.H6("Specification of the genetic architecture in AEGIS"),
            dash.dcc.Markdown(
                children=get_specification_of_gen_arch(), mathjax=True, dangerously_allow_html=True
            ),  # Display Markdown with MathJax
        ]
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
