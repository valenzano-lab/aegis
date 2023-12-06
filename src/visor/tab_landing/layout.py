from dash import html, dcc
from aegis.help import config
from visor import funcs
import yaml
import pathlib

# # TODO change text
# texts_domain = {
#     "recording": "Change which data are recorded and with what frequency.",
#     "predation": "asdf",
#     "computation": "wer",
#     "genetics": "asdf",
#     "initialization": "wer",
#     "infection": "asdf",
#     "ecology": "wer",
#     "environment": "asdf",
# }

# assert set(config.get_domains()) == set(
#     texts_domain
# ), "Specified and expected domains do not match"

# preface = [
#     html.Div(
#         children=[
#             # TODO change text
#             html.P(
#                 [
#                     """
#         Using this tab you can customize your simulation and run it.
#         Change the parameter values under the column name VALUE.
#         Run the simulation by giving it a unique id name and clicking the button 'run simulation'.
#         """
#                 ]
#             )
#         ],
#         style={"color": "white"},
#     )
# ]

# header = html.Tr(
#     [
#         html.Th("PARAMETER", style={"padding-left": "1.2rem"}),
#         html.Th("VALUE"),
#         html.Th("TYPE"),
#         html.Th("RANGE", className="valid-values"),
#         # html.Th("DOMAIN"),
#         html.Th("DESCRIPTION", style={"padding-right": "1.2rem"}),
#     ],
# )


@funcs.log_info
def get_landing_layout():
    doc_path = pathlib.Path(__file__).parent.parent.parent.parent / "documentation.yml"

    with open(doc_path, "r") as file_:
        doc = yaml.safe_load(file_)

    # # Group parameters by domain
    # subsets = {domain: [] for domain in texts_domain.keys()}
    # for param in config.params.values():
    #     subsets[param.domain].append(param)

    # # Generate layout

    # tables = []
    # for domain, subset in subsets.items():
    #     tables.extend(
    #         [
    #             html.Div(
    #                 children=[
    #                     html.P(domain, className="config-domain"),
    #                     html.P(texts_domain[domain], className="config-domain-desc"),
    #                 ],
    #             ),
    #             get_table(subset),
    #         ]
    #     )

    print(doc)

    layout = html.Div(
        id="landing-section",
        children=[
            html.H3("simulator of life history evolution"),
            html.P(
                # TODO not great; how evolution is shaped; second derivative
                """aegis is a scientific and educational tool for studying how
                    evolution of age-specific survival and reproduction rates is shaped by diverse factors –
                    demographic constraints (e.g. carrying capacity, population dynamics),
                    mortality factors (e.g. predation, infection, environmental hazards),
                    basic biology (e.g. mutation rate, recombination rate, mode of reproduction),
                    and other life history traits (e.g. maximum lifespan, age to maturity, age at menopause)."""
            ),
        ],
    )

    # # Generate layout
    # return html.Div(
    #     id="landing-section",
    #     children=[html.P(p) for p in doc["summary"]]
    #     + [html.P(doc["landing_page"]["oneliner"]), html.P(doc["landing_page"]["summary"])],
    #     # children=preface + tables,
    # )
    return layout


# @funcs.log_info
# def get_row(v):
#     if v.resrange_info:
#         resrange_info_message = (
#             f"Allowed parameter range for the server is {v.resrange_info}."
#         )
#     else:
#         resrange_info_message = ""

#     return html.Tr(
#         [
#             # PARAMETER
#             html.Td(v.get_name(), style={"padding-left": "1.2rem"}),
#             # VALUE
#             html.Td(
#                 children=dcc.Input(
#                     type="text",
#                     placeholder=str(v.default) if v.default is not None else "",
#                     # id=f"config-{v.key}",
#                     id={"type": "config-input", "index": v.key},
#                     autoComplete="off",
#                     className="config-input-class",
#                 ),
#             ),
#             # TYPE
#             html.Td(
#                 children=html.Label(
#                     v.dtype.__name__,
#                     className=f"dtype-{v.dtype.__name__} dtype",
#                 )
#             ),
#             # RANGE
#             html.Td(children=v.drange, className="data-range"),
#             # DESCRIPTION
#             html.Td(
#                 children=[
#                     v.info + ".",
#                     html.P(
#                         resrange_info_message,
#                         className="resrange_info_message",
#                     ),
#                 ],
#                 className="td-info",
#                 style={"padding-right": "0.8rem"},
#             ),
#         ],
#     )


# @funcs.log_info
# def get_table(params_subset):
#     return html.Table(
#         className="config-table",
#         children=[header]
#         + [get_row(v) for v in params_subset if not isinstance(v.default, list)],
#     )
