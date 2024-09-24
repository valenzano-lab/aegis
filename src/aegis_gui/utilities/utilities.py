import dash
import yaml

from aegis_sim.utilities.container import Container
from aegis_gui.guisettings.GuiSettings import gui_settings


def read_yml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_container(filename):
    return Container(gui_settings.sim_dir / filename)


def get_config_path(filename):
    return gui_settings.sim_dir / f"{filename}.yml"


def get_sim_paths(sim_dir=None, sort=True):
    if sim_dir is None:
        sim_dir = gui_settings.sim_dir
    paths = [p for p in sim_dir.iterdir() if p.is_dir()]
    if sort:
        paths = sorted(paths, key=lambda path: path.name)
    return paths


def get_sims():
    return [p.stem for p in get_sim_paths()]


def sim_exists(filename: str) -> bool:
    paths = get_sim_paths()
    return any(path.stem == filename for path in paths)


def get_icon(icon_name):
    return dash.html.Img(
        src=f"/aegis/assets/icons/{icon_name}.svg",
        width="16",
        height="16",
    )
