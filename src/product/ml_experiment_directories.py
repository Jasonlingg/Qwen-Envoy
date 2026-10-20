"""Open selected POSIX directories without following any ancestor symlink."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def open_directory(path: Path, *, create: bool = False):
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Selected directory must be absolute without parent traversal")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass  # The no-follow open below validates any existing entry.
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)
