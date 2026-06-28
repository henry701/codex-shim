from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

_MAX_PATH_SEGMENT_LEN = 200
_CACHE_VERSION = 1
_MAX_CACHED_RESPONSES_PER_SESSION = 1024

_UNSAFE_SEGMENT = re.compile(r"[^\w.\-]+")
_UNSCOPED_SESSION_KEY = "_unscoped"
_unscoped_warned = False


def session_key_from_headers(headers: Mapping[str, str]) -> str:
    global _unscoped_warned
    for name in ("session-id", "thread-id"):
        raw = headers.get(name)
        if isinstance(raw, str) and raw.strip():
            return sanitize_path_segment(raw.strip())
    if not _unscoped_warned:
        _unscoped_warned = True
        print(
            "[chatgpt-cache] request missing session-id and thread-id; using _unscoped partition",
            flush=True,
        )
    return _UNSCOPED_SESSION_KEY


def sanitize_path_segment(value: str) -> str:
    cleaned = value.replace("/", "_").replace("\\", "_").replace("..", "_")
    cleaned = _UNSAFE_SEGMENT.sub("_", cleaned).strip("._")
    if len(cleaned) > _MAX_PATH_SEGMENT_LEN:
        digest = hashlib.sha256(value.encode()).hexdigest()[:16]
        cleaned = f"{cleaned[:_MAX_PATH_SEGMENT_LEN - 17]}_{digest}"
    if not cleaned:
        return hashlib.sha256(value.encode()).hexdigest()[:32]
    return cleaned


def sanitize_response_filename(response_id: str) -> str:
    return f"{sanitize_path_segment(response_id)}.json"


class ChatgptConversationCache:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser()
        self._read_cache: dict[tuple[str, str], list[Any]] = {}

    def get(self, session_key: str, response_id: str) -> list[Any] | None:
        if not response_id:
            return None
        key = (session_key, response_id)
        cached = self._read_cache.get(key)
        if cached is not None:
            return copy.deepcopy(cached)
        path = self._entry_path(session_key, response_id)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        items = payload.get("items")
        if not isinstance(items, list) or not items:
            return None
        stored = copy.deepcopy(items)
        self._read_cache[key] = stored
        return copy.deepcopy(stored)

    def put(
        self,
        session_key: str,
        response_id: str,
        items: list[Any],
        *,
        terminal: bool = True,
    ) -> None:
        if not response_id or not items:
            return
        key = (session_key, response_id)
        self._read_cache[key] = copy.deepcopy(items)
        if not terminal:
            return
        path = self._entry_path(session_key, response_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            self._evict_if_needed(session_key)
            return
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = {"version": _CACHE_VERSION, "items": copy.deepcopy(items)}
        data = json.dumps(payload, separators=(",", ":"))
        tmp.write_text(data)
        os.replace(tmp, path)
        self._evict_if_needed(session_key)

    def stats(self) -> dict[str, int]:
        session_dirs = 0
        file_count = 0
        total_bytes = 0
        if self.root.is_dir():
            for child in self.root.iterdir():
                if not child.is_dir():
                    continue
                session_dirs += 1
                for path in child.glob("*.json"):
                    if not path.is_file():
                        continue
                    file_count += 1
                    try:
                        total_bytes += path.stat().st_size
                    except OSError:
                        pass
        return {
            "session_dirs": session_dirs,
            "file_count": file_count,
            "total_bytes": total_bytes,
            "read_cache_entries": len(self._read_cache),
        }

    def _entry_path(self, session_key: str, response_id: str) -> Path:
        session_dir = sanitize_path_segment(session_key)
        return self.root / session_dir / sanitize_response_filename(response_id)

    def _evict_if_needed(self, session_key: str) -> None:
        session_dir = self.root / sanitize_path_segment(session_key)
        if not session_dir.is_dir():
            return
        files = [path for path in session_dir.glob("*.json") if path.is_file()]
        overflow = len(files) - _MAX_CACHED_RESPONSES_PER_SESSION
        if overflow <= 0:
            return
        files.sort(key=lambda path: path.stat().st_mtime)
        for path in files[:overflow]:
            try:
                path.unlink()
            except OSError:
                pass
            self._read_cache.pop((session_key, self._response_id_from_filename(path.name)), None)

    @staticmethod
    def _response_id_from_filename(name: str) -> str:
        if name.endswith(".json"):
            return name[: -len(".json")]
        return name
