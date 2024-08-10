import subprocess


def get_folder_size_with_du(folder_path):
    result = subprocess.run(["du", "-sh", folder_path], stdout=subprocess.PIPE, text=True)
    return result.stdout.split()[0] + "B"
