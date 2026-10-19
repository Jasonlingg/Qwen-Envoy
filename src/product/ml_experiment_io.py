"""Bounded JSON and atomic local files for experiment storage; no model execution."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import stat
import threading
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

MAX_BYTES = 256 * 1_024
_LOCKS: dict[Path, threading.RLock] = {}
_GUARD = threading.Lock()
_NAME = re.compile(r"(?:experiment_[0-9a-f]{32}|object_[0-9a-f]{64})\.json")


def _tree(value) -> None:
    stack, count, characters = [(value, 0)], 0, 0
    while stack:
        item, depth = stack.pop()
        count += 1
        if depth > 20 or count > 20_000 or characters > MAX_BYTES:
            raise ValueError("Experiment JSON exceeds depth or element limits")
        if isinstance(item, dict):
            if any(type(key) is not str for key in item):
                raise ValueError("Experiment JSON keys must be strings")
            characters += sum(len(key) for key in item)
            if len(item) > 20_000:
                raise ValueError("Experiment JSON has too many entries")
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            if len(item) > 20_000:
                raise ValueError("Experiment JSON has too many entries")
            stack.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            characters += len(item)
        elif (type(item) not in (str, bool, int, float, type(None))
              or type(item) is float and not math.isfinite(item)):
            raise ValueError("Experiment values must be finite JSON data")
    if characters > MAX_BYTES:
        raise ValueError("Experiment JSON exceeds 256 KiB")


def encode(value) -> bytes:
    _tree(value)
    try:
        raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("Invalid experiment JSON") from exc
    if len(raw) > MAX_BYTES:
        raise ValueError("Experiment JSON exceeds 256 KiB")
    return raw


def _pairs(items):
    value = {}
    for key, item in items:
        if key in value:
            raise ValueError("Duplicate experiment JSON key")
        value[key] = item
    return value


def decode(raw: bytes):
    if len(raw) > MAX_BYTES:
        raise ValueError("Experiment JSON exceeds 256 KiB")
    try:
        value = json.loads(raw, object_pairs_hook=_pairs)
        _tree(value)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("Invalid experiment JSON") from exc
    return value


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class LocalFiles:
    def __init__(self, root: Path):
        path = Path(root).expanduser().absolute()
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise ValueError("Experiment store must not use symlinks")
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.root = path.resolve(strict=True)
        with _GUARD:
            self.thread_lock = _LOCKS.setdefault(self.root, threading.RLock())

    @contextmanager
    def locked(self):
        try:
            with self.thread_lock:
                directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    lock = os.open(".experiments.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW
                                   | os.O_NONBLOCK, 0o600, dir_fd=directory)
                    try:
                        if not stat.S_ISREG(os.fstat(lock).st_mode):
                            raise ValueError("Invalid experiment store lock")
                        fcntl.flock(lock, fcntl.LOCK_EX)
                        yield directory
                    finally:
                        os.close(lock)
                finally:
                    os.close(directory)
        except OSError as exc:
            raise ValueError("Experiment store is inaccessible or unsafe") from exc

    def read(self, directory: int, name: str) -> bytes:
        if not _NAME.fullmatch(name):
            raise ValueError("Invalid experiment storage name")
        try:
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                 dir_fd=directory)
            with os.fdopen(descriptor, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError("Experiment record is not a regular file")
                raw = stream.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise ValueError("Experiment record exceeds 256 KiB")
            return raw
        except OSError as exc:
            raise ValueError("Experiment record is missing or unsafe") from exc

    def object(self, directory: int, value: dict) -> str:
        raw = encode(value)
        sha = digest(raw)
        self.write(directory, f"object_{sha}.json", raw, immutable=True)
        return sha

    def read_object(self, directory: int, sha: str) -> dict:
        raw = self.read(directory, f"object_{sha}.json")
        if digest(raw) != sha:
            raise ValueError("Experiment object hash mismatch")
        return decode(raw)

    def write(self, directory: int, name: str, raw: bytes, *, immutable=False) -> None:
        if not _NAME.fullmatch(name):
            raise ValueError("Invalid experiment storage name")
        temporary = f".experiment-{uuid4().hex}.tmp"
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                 0o600, dir_fd=directory)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            if immutable:
                try:
                    os.link(temporary, name, src_dir_fd=directory, dst_dir_fd=directory,
                            follow_symlinks=False)
                except FileExistsError:
                    if self.read(directory, name) != raw:
                        raise ValueError("Existing experiment object differs") from None
            else:
                os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
            os.fsync(directory)
        finally:
            try:
                os.unlink(temporary, dir_fd=directory)
            except FileNotFoundError:
                pass  # Atomic replacement already consumed the temporary file.
