import re
import psutil
import logging
from typing import Set


def run_ps_af() -> Set[str]:
    """Returns names of simulations whose processes are currently running based on the OS."""
    config_files = set()
    try:
        # Define the pattern for matching process command lines
        pattern = re.compile(r"python3 -m aegis --config_path .*/(.*)\.yml")

        # Iterate through all running processes
        for proc in psutil.process_iter(["cmdline"]):
            try:
                # Get the command line arguments of the process
                cmdline = " ".join(proc.info["cmdline"])

                # Match the command line with the pattern
                match = pattern.search(cmdline)
                if match:
                    config_file = match.group(1)
                    config_files.add(config_file)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process has been terminated or we don't have permission to access it
                continue

    except Exception as e:
        logging.error(f"An error occurred: {e}")

    return config_files
