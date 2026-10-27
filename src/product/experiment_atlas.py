"""Read-only, bounded access to the curated historical experiment atlas."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from pathlib import Path

_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_LINK = re.compile(r"\[([^]]+)\]\(([^)]+)\)")
_WORD = re.compile(r"[a-z0-9]+")
_MAX_ATLAS_BYTES = 128_000
_MAX_SOURCE_BYTES = 4_000_000


class ExperimentAtlas:
    """Expose saved experiment summaries without importing or changing their runs."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.document = self.root / "docs" / "EXPERIMENT_ATLAS.md"

    def _entries(self):
        raw = self.document.read_bytes()
        if len(raw) > _MAX_ATLAS_BYTES:
            raise ValueError("Experiment atlas exceeds size limit")
        lines = raw.decode("utf-8").splitlines()
        try:
            header = lines.index(
                "| ID | Question and actual attempt | Saved observation | "
                "Decision and next question | Primary evidence |")
        except ValueError as exc:
            raise ValueError("Experiment atlas table is missing") from exc
        entries = []
        for line in lines[header + 2:]:
            if not line.startswith("| "):
                break
            fields = [part.strip() for part in line.strip().strip("|").split("|")]
            if len(fields) != 5:
                raise ValueError("Malformed experiment atlas row")
            atlas_id = fields[0].strip("`")
            if not _ID.fullmatch(atlas_id) or any(e["atlas_id"] == atlas_id for e in entries):
                raise ValueError("Invalid or repeated experiment atlas ID")
            sources = []
            for label, target in _LINK.findall(fields[4]):
                if target.startswith(("https://", "http://")):
                    raise ValueError("Atlas source must be a local saved artifact")
                relative = target.split("#", 1)[0]
                path = self._source_path(relative)
                sources.append({"label": label, "path": path.relative_to(self.root).as_posix(),
                                "anchor": target.split("#", 1)[1] if "#" in target else None})
            if not sources:
                raise ValueError("Experiment atlas row has no saved source")
            entries.append({"atlas_id": atlas_id, "attempt": fields[1],
                            "observation": fields[2], "decision": fields[3],
                            "sources": sources})
        return entries, hashlib.sha256(raw).hexdigest()

    def _source_path(self, relative: str) -> Path:
        if not relative or relative.startswith("/"):
            raise ValueError("Invalid atlas source path")
        candidate = self.document.parent / relative
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("Atlas source escapes repository") from exc
        if any(part.is_symlink() for part in (candidate, *candidate.parents)
               if part == self.root or self.root in part.parents):
            raise ValueError("Atlas source may not use symlinks")
        if not resolved.is_file():
            raise ValueError("Atlas source is missing")
        return resolved

    def find(self, query: str | None = None, limit: int = 12) -> dict:
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError("Limit must be an integer in 1..20")
        if query is not None and (not isinstance(query, str)
                                  or not 2 <= len(query.strip()) <= 200):
            raise ValueError("Query must contain 2..200 characters")
        entries, atlas_sha = self._entries()
        terms = set(_WORD.findall(query.lower())) if query else set()
        if query and not terms:
            raise ValueError("Query needs a searchable term")
        matches = []
        for entry in entries:
            words = set(_WORD.findall(" ".join([entry["atlas_id"], entry["attempt"],
                entry["observation"], entry["decision"]]).lower()))
            score = len(words & terms) if terms else 0
            if terms and not score:
                continue
            matches.append({"atlas_id": entry["atlas_id"], "attempt": entry["attempt"],
                            "observation": entry["observation"], "decision": entry["decision"],
                            "score": score, "source_count": len(entry["sources"])})
        matches.sort(key=lambda row: (-row["score"], row["atlas_id"]))
        return {"query": query, "match_kind": "local_word_overlap",
                "atlas_sha256": atlas_sha, "total_entries": len(entries),
                "matches": matches[:limit], "truncated": len(matches) > limit,
                "warning": "Curated historical summary; inspect sources before relying on claims."}

    def get(self, atlas_id: str) -> dict:
        if not isinstance(atlas_id, str) or not _ID.fullmatch(atlas_id):
            raise ValueError("Invalid experiment atlas ID")
        entries, atlas_sha = self._entries()
        entry = next((item for item in entries if item["atlas_id"] == atlas_id), None)
        if entry is None:
            raise ValueError("Unknown experiment atlas ID")
        return {**entry, "atlas_sha256": atlas_sha,
                "warning": "The summary is curated; source bytes and semantic claims need review."}

    def read_source(self, atlas_id: str, source_index: int,
                    start: int = 0, length: int = 4_000) -> dict:
        entry = self.get(atlas_id)
        if type(source_index) is not int or not 0 <= source_index < len(entry["sources"]):
            raise ValueError("Invalid atlas source index")
        if type(start) is not int or start < 0:
            raise ValueError("Source start must be nonnegative")
        if type(length) is not int or not 1 <= length <= 8_000:
            raise ValueError("Source length must be in 1..8000")
        source = entry["sources"][source_index]
        path = self._source_path("../" + source["path"])
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_SOURCE_BYTES:
                raise ValueError("Atlas source must be a bounded regular file")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                sha = hashlib.sha256()
                for chunk in iter(lambda: stream.read(64 * 1024), b""):
                    sha.update(chunk)
                stream.seek(min(start, info.st_size))
                excerpt = stream.read(length)
        finally:
            os.close(fd)
        return {"atlas_id": atlas_id, "source_index": source_index,
                "path": source["path"], "label": source["label"],
                "sha256": sha.hexdigest(), "file_bytes": info.st_size,
                "start": start, "text": excerpt.decode("utf-8", errors="replace"),
                "truncated": start + len(excerpt) < info.st_size,
                "warning": "Exact bytes establish provenance, not semantic support."}
