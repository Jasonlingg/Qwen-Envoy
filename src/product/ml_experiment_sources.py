"""Read only explicitly selected JSON exports/cache files, without following symlinks."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from src.product.ml_experiment_directories import open_directory
from src.product.ml_experiment_io import decode

MAX_SOURCE_BYTES = 2 * 1_024 * 1_024


def read_json(root: Path, relative: str):
    root = Path(root).expanduser().absolute()
    parts = relative.split("/")
    if (any(p in {"", ".", ".."} for p in parts) or "\\" in relative
            or any(ord(char) < 32 for char in relative)
            or any(p.is_symlink() for p in (root, *root.parents))):
        raise ValueError("Source must remain inside the selected directory without symlinks")
    descriptors = []
    try:
        with open_directory(root) as directory:
            for part in parts[:-1]:
                directory = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                    dir_fd=directory)
                descriptors.append(directory)
            descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                 dir_fd=directory)
            with os.fdopen(descriptor, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError("Selected source is not a regular file")
                raw = stream.read(MAX_SOURCE_BYTES + 1)
        return raw, decode(raw, max_bytes=MAX_SOURCE_BYTES)
    except OSError as exc:
        raise ValueError("Selected JSON source is unavailable or unsafe") from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def select_pointer(value, pointer: str):
    """RFC 6901 selection with strict array indexes, not Python indexing rules."""
    if pointer == "":
        return value
    if not pointer.startswith("/") or any(
            not part.startswith(("0", "1")) for part in pointer.split("~")[1:]):
        raise ValueError("Invalid JSON Pointer")
    for raw in pointer[1:].split("/"):
        key = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict) and key in value:
            value = value[key]
        elif (isinstance(value, list) and key.isascii() and key.isdecimal()
              and (key == "0" or not key.startswith("0")) and len(key) <= 10
              and int(key) < len(value)):
            value = value[int(key)]
        else:
            raise ValueError("JSON Pointer does not select a source value")
    return value
