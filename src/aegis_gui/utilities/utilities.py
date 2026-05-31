import dash
import re
import yaml

from aegis_sim.utilities.container import Container
from aegis_gui.guisettings.GuiSettings import gui_settings


# Strict whitelist for sim names. Filesystem-safe characters only — no slashes,
# no leading dots (would hide files), no whitespace, capped length.
_SIM_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")


class UnsafeSimNameError(ValueError):
    """Raised when a sim name fails the strict whitelist or escapes sim_dir."""


def safe_sim_name(name) -> str:
    """Validate `name` as a safe simulation identifier.

    Accepts letters, digits, underscore, dash, and dot (not leading), max 200
    chars. Rejects anything else — including slashes, traversal sequences,
    whitespace, null bytes. Returns the validated string.
    """
    if not isinstance(name, str):
        raise UnsafeSimNameError(f"sim name must be str, got {type(name).__name__}")
    if not _SIM_NAME_PATTERN.fullmatch(name):
        raise UnsafeSimNameError(f"rejected sim name: {name!r}")
    return name


def safe_sim_path(name) -> "pathlib.Path":
    """Return the validated directory path for a sim, guaranteed to be inside
    `gui_settings.sim_dir`. Raises UnsafeSimNameError if the resolved path
    escapes the sim directory (defense in depth even after the regex check).
    """
    import pathlib

    name = safe_sim_name(name)
    base = pathlib.Path(gui_settings.sim_dir).resolve()
    candidate = (base / name).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        raise UnsafeSimNameError(f"sim path escapes sim_dir: {name!r}")
    return candidate


def read_yml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_container(filename):
    safe_sim_name(filename)
    return Container(gui_settings.sim_dir / filename)


def get_config_path(filename):
    safe_sim_name(filename)
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
