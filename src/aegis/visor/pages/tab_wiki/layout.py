import dash
from dash import html, dash_table
from aegis.visor.utilities.utilities import OUTPUT_SPECIFICATIONS
from aegis.visor.utilities import log_funcs

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


layout = html.Div(
    [
        html.P("""This is the wiki tab."""),
        make_output_specification_table(),
    ]
)
