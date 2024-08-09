"""Keeping track of running simulation processes"""

import subprocess
import re
import logging
from typing import Set

# TODO test this on different setups; is it picking up the python process line?


def run_ps_af() -> Set[str]:
    """Returns names of simulations whose processes are currently running"""
    config_files = set()
    try:
        # Run the 'ps -af' command
        result = subprocess.run(["ps", "-af"], capture_output=True, text=True, check=True)

        # Process the output to find lines matching the pattern
        for line in result.stdout.splitlines():
            match = re.search(r"python3 -m aegis --config_path .*/(.*)\.yml", line)
            if match:
                config_file = match.group(1)
                config_files.add(config_file)
    except subprocess.CalledProcessError as e:
        logging.error(f"An error occurred: {e}")

    return config_files
