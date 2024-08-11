import math
import os


def get_folder_size_with_du(folder_path):
    """Returns the folder size for Linux, macOS, and Windows."""

    size = get_folder_size_python(folder_path)
    return size


def convert_size(size_bytes):
    """Converts bytes to a human-readable size format (e.g., KB, MB, GB)."""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


def get_folder_size_python(folder_path):
    """Returns the total size of the folder in a human-readable format."""
    total_size = 0

    # Traverse the directory tree
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            # Get the full file path
            filepath = os.path.join(dirpath, filename)
            # Skip if it's a symbolic link
            if not os.path.islink(filepath):
                # Accumulate file size
                total_size += os.path.getsize(filepath)

    return convert_size(total_size)
