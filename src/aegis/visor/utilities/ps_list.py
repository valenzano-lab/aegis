import subprocess
import re
from typing import Set


def run_ps_af() -> Set[str]:
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
        print(f"An error occurred: {e}")

    return config_files


if __name__ == "__main__":
    config_files = run_ps_af()
    print(f"Found config files: {config_files}")
