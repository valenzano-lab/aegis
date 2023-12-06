# from dash import html, dcc, callback, Output, Input, State, ALL, MATCH, ctx
# from visor import funcs
# from aegis.help import config


# @callback(
#     Output("config-make-text", "value"),
#     Input("simulation-run-button", "n_clicks"),
#     State("config-make-text", "value"),
#     State({"type": "config-input", "index": ALL}, "value"),
#     prevent_initial_call=True,
# )
# @funcs.log_info
# def run_simulation(n_clicks, filename, values):
#     if n_clicks is None:
#         return

#     # make config file
#     default_config = config.get_default_parameters()
#     custom_config = {
#         k: val
#         for (k, v), val in zip(default_config.items(), values)
#         if not isinstance(v, list)
#     }
#     funcs.make_config_file(filename, custom_config)

#     # run simulation
#     funcs.run(filename)

#     return ""


# @callback(
#     Output("simulation-run-button", "disabled"),
#     Input("config-make-text", "value"),
#     Input({"type": "config-input", "index": ALL}, "value"),
# )
# @funcs.log_info
# def block_sim_button(filename, values):
#     default_config = config.get_default_parameters()
#     for k, value in zip(default_config, values):
#         if value != "" and value is not None:
#             param = config.params[k]
#             val = param.convert(value)
#             valid = param.resrange(val)
#             if not valid:
#                 return True

#     if filename is None or filename == "" or "." in filename:
#         return True

#     sim_exists = funcs.sim_exists(filename)
#     if sim_exists:
#         return True
#     return False


# @callback(
#     # Output("simulation-run-button", "disabled"),
#     # Input("config-make-text", "value"),
#     Output({"type": "config-input", "index": MATCH}, "className"),
#     Input({"type": "config-input", "index": MATCH}, "value"),
#     State({"type": "config-input", "index": MATCH}, "className"),
#     prevent_initial_call=True,
# )
# @funcs.log_info
# def block_config_input(value, className):
#     print(className, "a")
#     if ctx.triggered_id is None:
#         return className

#     param = config.params[ctx.triggered_id["index"]]

#     className = className.replace(" disabled", "")

#     inside_range = param.resrange(param.convert(value))
#     if not inside_range:
#         className += " disabled"

#     print(className, "a")
#     return className
