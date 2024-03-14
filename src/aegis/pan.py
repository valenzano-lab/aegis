import pathlib
import time
import numpy as np
import shutil
import yaml
import types

from aegis.help.config import get_default_parameters, validate


cnf = types.SimpleNamespace()
var = types.SimpleNamespace()
rng = None


def set_up_cnf(custom_config_path, running_on_server):
    global cnf

    # Read config parameters from the custom config file
    if custom_config_path == "":
        custom_config_params = {}
    else:
        with open(custom_config_path, "r") as f:
            custom_config_params = yaml.safe_load(f)
        if custom_config_params is None:  # If config file is empty
            custom_config_params = {}

    # Read config parameters from the default config file
    default_config_params = get_default_parameters()

    # Fuse
    params = {}
    params.update(default_config_params)
    params.update(custom_config_params)

    validate(params, validate_resrange=running_on_server)

    for k, v in params.items():
        setattr(cnf, k, v)


def set_up_var():
    global var

    if cnf.RANDOM_SEED is None:
        random_seed = np.random.randint(1, 10**6)
    else:
        random_seed = cnf.RANDOM_SEED

    var.stage = 1
    var.random_seed = random_seed
    global rng
    rng = np.random.default_rng(random_seed)


def skip(rate):
    """Should you skip an action performed at a certain rate"""

    # Skip if rate deactivated
    if rate <= 0:
        return True

    # Do not skip first stage
    if var.stage == 1:
        return False

    # Skip unless stage is divisible by rate
    return var.stage % rate > 0


# Decorators
def profile_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000
        print(f"{func.__name__} took {execution_time:.6f} ms to execute")
        return result

    return wrapper


# INITIALIZE


def init(config_path, pickle_path_, overwrite):

    # config_path, pickle_path, overwrite = parse()

    global output_path
    global time_start
    global here
    global progress_path
    global pickle_path

    pickle_path = pickle_path_

    time_start = time.time()
    here = pathlib.Path(__file__).absolute().parent

    config_path = pathlib.Path(config_path)

    set_up_cnf(config_path, running_on_server=False)
    set_up_var()
    # Output directory
    output_path = config_path.parent / config_path.stem
    if output_path.exists() and output_path.is_dir():
        if overwrite:
            shutil.rmtree(output_path)  # Delete previous directory if existing
        else:
            raise Exception(f"{output_path} already exists. To overwrite, add flag --overwrite or -o.")
    output_path.mkdir(parents=True, exist_ok=True)

    # Set up progress log
    progress_path = output_path / "progress.log"
    content = ("stage", "ETA", "t1M", "runtime", "stg/min", "popsize")
    with open(progress_path, "wb") as f:
        np.savetxt(f, [content], fmt="%-10s", delimiter="| ")
