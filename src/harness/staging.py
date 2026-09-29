"""Atomically stage agent-authored Markdown drafts inside an Obsidian inbox.

This module has no promotion operation. Callers choose a *run ID*, not an output
directory; the only published destination is ``<vault>/_inbox/<run_id>``.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
import sys
import unicodedata
from ctypes import CDLL, c_char_p, c_int, c_uint, get_errno
from dataclasses import dataclass
from errno import EEXIST, ENOTEMPTY
from pathlib import Path
from typing import Mapping

import yaml

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")
_UNSAFE_NAME_CHARS = set('\\:<>?*|"')
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_APPROVAL_KEYS = {
    "approved",
    "approved_by",
    "approved_by_user",
    "approval",
    "human_approval",
    "human_reviewed",
    "reviewed_by",
    "reviewed_by_user",
    "user_approved",
    "user_approval",
    "user_reviewed",
}
_APPROVAL_VALUES = {
    "approved",
    "approved_by_user",
    "human_reviewed",
    "reviewed_by_user",
    "user_approved",
    "user_reviewed",
}


class StagingError(ValueError):
    """An input or vault layout is unsafe for staging."""


class DuplicateRunError(FileExistsError):
    """A run ID already has a staged batch in this vault."""


class PublishedButUnconfirmedError(OSError):
    """A published batch is visible, but its directory fsync failed."""

    def __init__(self, run_dir: Path):
        super().__init__(f"staged batch is visible but durability is unconfirmed: {run_dir}")
        self.run_dir = run_dir
        self.manifest_path = run_dir / "manifest.json"


@dataclass(frozen=True)
class StageResult:
    run_id: str
    run_dir: Path
    files: Mapping[str, Path]
    hashes: Mapping[str, str]
    manifest_path: Path


class _UniqueKeysLoader(yaml.SafeLoader):
    """Keep contradictory duplicate frontmatter keys from being interpreted differently."""


def _construct_unique_mapping(
    loader: _UniqueKeysLoader, node: yaml.MappingNode
) -> dict[str, object]:
    mapping: dict[str, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str) or key in mapping:
            raise StagingError("frontmatter keys must be unique strings")
        mapping[key] = loader.construct_object(value_node, deep=True)
    return mapping


_UniqueKeysLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _review_marker(value: str) -> str:
    return re.sub(r"[\s-]+", "_", value.strip().casefold())


def _has_approval_marker(value: object, seen: set[int] | None = None) -> bool:
    if seen is None:
        seen = set()
    if isinstance(value, Mapping):
        if id(value) in seen:
            return False
        seen.add(id(value))
        return any(
            _review_marker(key) in _APPROVAL_KEYS or _has_approval_marker(child, seen)
            for key, child in value.items()
        )
    if isinstance(value, list):
        if id(value) in seen:
            return False
        seen.add(id(value))
        return any(_has_approval_marker(child, seen) for child in value)
    return isinstance(value, str) and _review_marker(value) in _APPROVAL_VALUES


def _validate_draft_content(name: str, content: str) -> None:
    lines = content.splitlines()
    if not lines or lines[0] != "---":
        raise StagingError(f"draft must start with YAML frontmatter: {name!r}")
    try:
        closing = lines.index("---", 1)
    except ValueError as exc:
        raise StagingError(f"draft frontmatter is not closed: {name!r}") from exc
    frontmatter_text = "\n".join(lines[1:closing])
    if len(frontmatter_text.encode("utf-8")) > 64_000:
        raise StagingError(f"draft frontmatter is too large: {name!r}")
    try:
        frontmatter = yaml.load(frontmatter_text, Loader=_UniqueKeysLoader)
    except yaml.YAMLError as exc:
        raise StagingError(f"invalid YAML frontmatter: {name!r}") from exc
    if not isinstance(frontmatter, Mapping):
        raise StagingError(f"draft frontmatter must be a mapping: {name!r}")
    if frontmatter.get("review_status") != "agent_authored_draft":
        raise StagingError(f"draft must declare review_status: agent_authored_draft: {name!r}")
    if _has_approval_marker(frontmatter):
        raise StagingError(f"draft cannot declare human or user approval: {name!r}")


def _validate_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise StagingError(
            "run_id must be 1-80 ASCII letters, digits, '_' or '-', starting with a letter or digit"
        )


def _validate_file_path(name: str) -> tuple[str, ...]:
    if not isinstance(name, str) or not name or name.startswith("/"):
        raise StagingError("draft paths must be nonempty relative POSIX paths")
    parts = tuple(name.split("/"))
    for part in parts:
        if (
            part in ("", ".", "..")
            or part.startswith(".")
            or part.endswith((".", " "))
            or any(char in _UNSAFE_NAME_CHARS or ord(char) < 32 for char in part)
        ):
            raise StagingError(f"unsafe draft path: {name!r}")
    if not parts[-1].lower().endswith(".md"):
        raise StagingError(f"draft must be a Markdown file: {name!r}")
    if parts[0].casefold() == "manifest.json":
        raise StagingError("manifest.json is reserved for the staging manifest")
    if any(part.lower().endswith(".md") for part in parts[:-1]):
        raise StagingError(f"a Markdown file cannot be a parent directory: {name!r}")
    return parts


def _validated_files(files: Mapping[str, str]) -> dict[str, tuple[tuple[str, ...], bytes]]:
    if not isinstance(files, Mapping) or not files:
        raise StagingError("files must be a nonempty mapping of Markdown paths to text")
    validated: dict[str, tuple[tuple[str, ...], bytes]] = {}
    seen_components: dict[tuple[str, ...], tuple[str, ...]] = {}
    seen_files: set[tuple[str, ...]] = set()
    for name, content in files.items():
        parts = _validate_file_path(name)
        if not isinstance(content, str):
            raise StagingError(f"draft content must be text: {name!r}")
        _validate_draft_content(name, content)
        canonical_parts = tuple(unicodedata.normalize("NFC", part).casefold() for part in parts)
        for length in range(1, len(parts) + 1):
            canonical_prefix = canonical_parts[:length]
            raw_prefix = parts[:length]
            earlier = seen_components.setdefault(canonical_prefix, raw_prefix)
            if earlier != raw_prefix:
                raise StagingError(f"draft paths collide on case-insensitive filesystems: {name!r}")
        if canonical_parts in seen_files:
            raise StagingError(f"duplicate draft path: {name!r}")
        seen_files.add(canonical_parts)
        validated[name] = (parts, content.encode("utf-8"))
    return validated


def _write_bytes_at(parent_fd: int, name: str, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    fd = os.open(name, flags, 0o600, dir_fd=parent_fd)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _mkdir_and_open_at(parent_fd: int, name: str) -> int:
    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
    except FileExistsError:
        pass
    try:
        return os.open(name, _DIR_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        raise StagingError(f"unsafe directory in staging path: {name!r}") from exc


def _write_draft_at(temp_fd: int, parts: tuple[str, ...], data: bytes) -> None:
    current_fd = temp_fd
    opened: list[int] = []
    try:
        for part in parts[:-1]:
            current_fd = _mkdir_and_open_at(current_fd, part)
            opened.append(current_fd)
        _write_bytes_at(current_fd, parts[-1], data)
        for fd in reversed(opened):
            os.fsync(fd)
    finally:
        for fd in reversed(opened):
            os.close(fd)


def _remove_tree_at(parent_fd: int, name: str) -> None:
    """Clean an unpublished batch without following any inserted symlinks."""
    tree_fd = os.open(name, _DIR_FLAGS, dir_fd=parent_fd)
    try:
        with os.scandir(tree_fd) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    _remove_tree_at(tree_fd, entry.name)
                else:
                    os.unlink(entry.name, dir_fd=tree_fd)
    finally:
        os.close(tree_fd)
    os.rmdir(name, dir_fd=parent_fd)


def _publish_no_replace(inbox_fd: int, temp_name: str, run_id: str) -> None:
    """Atomically rename a complete batch, failing if *any* target already exists.

    Plain ``os.rename`` may replace an empty directory between a duplicate check
    and publication. macOS and Linux both expose a no-replace variant. On other
    platforms or filesystems, fail closed rather than using an unsafe fallback.
    """
    libc = CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        try:
            rename = libc.renameatx_np
        except AttributeError as exc:
            raise StagingError("atomic no-replace rename is unavailable") from exc
        rename.argtypes = [c_int, c_char_p, c_int, c_char_p, c_uint]
        rename.restype = c_int
        exclusive_flag = 0x00000004  # RENAME_EXCL, <sys/stdio.h>
    elif sys.platform.startswith("linux"):
        try:
            rename = libc.renameat2
        except AttributeError as exc:
            raise StagingError("atomic no-replace rename is unavailable") from exc
        rename.argtypes = [c_int, c_char_p, c_int, c_char_p, c_uint]
        rename.restype = c_int
        exclusive_flag = 1  # RENAME_NOREPLACE, <linux/fs.h>
    else:
        raise StagingError("atomic no-replace rename is unsupported on this platform")

    if rename(inbox_fd, os.fsencode(temp_name), inbox_fd, os.fsencode(run_id), exclusive_flag):
        error = get_errno()
        if error in (EEXIST, ENOTEMPTY):
            raise DuplicateRunError(f"run already staged: {run_id}")
        raise OSError(error, os.strerror(error))


def stage_drafts(vault: Path, run_id: str, files: Mapping[str, str]) -> StageResult:
    """Publish one complete batch under ``vault/_inbox/run_id``.

    The vault must already exist. ``_inbox`` is created if needed. A process-wide
    file lock serializes batches, and an unpublished private directory is renamed
    into place only after every draft and its hash manifest has been written.
    Existing run IDs fail rather than being replaced.
    """
    _validate_run_id(run_id)
    validated = _validated_files(files)
    vault = Path(vault)
    if vault.is_symlink() or not vault.is_dir():
        raise StagingError("vault must be an existing real directory")
    vault = vault.resolve(strict=True)
    run_dir = vault / "_inbox" / run_id

    vault_fd = os.open(vault, _DIR_FLAGS)
    try:
        inbox_fd = _mkdir_and_open_at(vault_fd, "_inbox")
        try:
            lock_fd = os.open(
                ".stage.lock",
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                0o600,
                dir_fd=inbox_fd,
            )
            try:
                if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
                    raise StagingError("_inbox/.stage.lock must be a regular file")
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                try:
                    try:
                        os.stat(run_id, dir_fd=inbox_fd, follow_symlinks=False)
                    except FileNotFoundError:
                        pass
                    else:
                        raise DuplicateRunError(f"run already staged: {run_id}")

                    temp_name = f"._stage-{run_id}-{secrets.token_hex(12)}"
                    os.mkdir(temp_name, 0o700, dir_fd=inbox_fd)
                    published = False
                    try:
                        temp_fd = os.open(temp_name, _DIR_FLAGS, dir_fd=inbox_fd)
                        try:
                            records: dict[str, dict[str, str | int]] = {}
                            for name in sorted(validated):
                                parts, data = validated[name]
                                _write_draft_at(temp_fd, parts, data)
                                records[name] = {
                                    "sha256": hashlib.sha256(data).hexdigest(),
                                    "bytes": len(data),
                                }
                            manifest = {
                                "schema_version": "staged-drafts-v1",
                                "run_id": run_id,
                                "files": records,
                            }
                            manifest_text = json.dumps(
                                manifest, indent=2, sort_keys=True, ensure_ascii=False
                            )
                            manifest_bytes = (manifest_text + "\n").encode("utf-8")
                            _write_bytes_at(temp_fd, "manifest.json", manifest_bytes)
                            os.fsync(temp_fd)
                        finally:
                            os.close(temp_fd)
                        _publish_no_replace(inbox_fd, temp_name, run_id)
                        published = True
                        try:
                            os.fsync(inbox_fd)
                        except OSError as exc:
                            raise PublishedButUnconfirmedError(run_dir) from exc
                    finally:
                        if not published:
                            _remove_tree_at(inbox_fd, temp_name)

                    return StageResult(
                        run_id=run_id,
                        run_dir=run_dir,
                        files={name: run_dir.joinpath(*validated[name][0]) for name in validated},
                        hashes={name: record["sha256"] for name, record in records.items()},
                        manifest_path=run_dir / "manifest.json",
                    )
                finally:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        finally:
            os.close(inbox_fd)
    finally:
        os.close(vault_fd)
