import dash
from dash import html, dash_table
from aegis_gui.utilities.utilities import OUTPUT_SPECIFICATIONS
from aegis_gui.utilities import log_funcs
import dash_bootstrap_components as dbc


dash.register_page(__name__, path="/wiki", name="wiki")


@log_funcs.log_debug
def make_output_specification_table():
    """

    Documentation for plotly datatables: https://dash.plotly.com/datatable
    """
    data = [
        {key: specs.get(key, "!!! nan") for key in OUTPUT_SPECIFICATIONS[0].keys()} for specs in OUTPUT_SPECIFICATIONS
    ]
    columns = [{"id": c, "name": c} for c in OUTPUT_SPECIFICATIONS[0].keys()]
    return dash_table.DataTable(
        data=data,
        columns=columns,
        style_data={"whiteSpace": "normal", "height": "auto"},
    )


def layout() -> html.Div:
    return html.Div(
        [
            html.H1("""Wiki tab"""),
            html.P("Specification of output files"),
            make_accordion(),
            # make_output_specification_table(),
        ]
    )


def make_accordion_item(d):

    return dbc.ListGroup(
        children=[dbc.ListGroupItem(children=[dash.html.Strong(f"{k}: "), v]) for k, v in d.items() if k != "path"],
        flush=True,
    )


def make_accordion():
    data = [
        {key: specs.get(key, "!!! nan") for key in OUTPUT_SPECIFICATIONS[0].keys()} for specs in OUTPUT_SPECIFICATIONS
    ]
    return dbc.Accordion(
        id="wiki-accordion",
        children=[dbc.AccordionItem(title=d["path"], children=make_accordion_item(d)) for d in data],
    )
