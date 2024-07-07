from dash import dash_table
from aegis.utilities.container import Container
import pathlib
import datetime
from aegis.utilities.get_folder_size import get_folder_size_with_du
from aegis.visor.utilities import utilities


def get_data(path: pathlib.Path):

    container = Container(path)
    log = container.get_log()
    input_summary = container.get_input_summary()
    output_summary = container.get_output_summary()
    basepath = container.basepath
    filename = container.basepath.stem
    ticker_stopped = container.has_ticker_stopped()

    if len(log) > 0:
        logline = log.iloc[-1].to_dict()
    else:
        logline = {"ETA": None, "step": None, "stg/min": None}

    if not output_summary:
        status = ["not finished", "not extinct"]
    elif output_summary["extinct"]:
        status = ["finished", "extinct"]
    else:
        status = ["finished", "not extinct"]

    time_of_creation = (
        datetime.datetime.fromtimestamp(input_summary["time_start"]).strftime("%Y-%m-%d %H:%M")
        if input_summary
        else None
    )

    time_of_finishing = (
        datetime.datetime.fromtimestamp(input_summary["time_start"]).strftime("%Y-%m-%d %H:%M")
        if output_summary
        else None
    )

    extinct = status[1]
    running = "no" if ticker_stopped else "yes"

    size = get_folder_size_with_du(basepath)

    eta = logline["ETA"]

    return {
        "id": filename,
        "created": time_of_creation,
        "finished": time_of_finishing,
        "extinct": extinct,
        "running": running,
        "size": size,
        "eta": eta if time_of_finishing is None else "       ",
        "filepath": str(path),
    }


def make_table():
    paths = utilities.get_sim_paths()
    datas = [get_data(path) for path in paths]

    return dash_table.DataTable(
        data=datas,
        columns=[{"name": col, "id": col} for col in datas[0].keys()],
    )
