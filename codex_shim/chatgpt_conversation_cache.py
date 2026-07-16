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
_DEFAULT_MAX_CACHE_BYTES = 512 * 1024 * 1024
_BYTE_SUFFIX_MULTIPLIERS = {"k": 1024, "m": 1024**2, "g": 1024**3}

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


def thread_id_from_headers(headers: Mapping[str, str]) -> str | None:
    """Codex Desktop thread id for subagent isolation.

    Prefer explicit ``thread-id``, then ``x-codex-turn-metadata.thread_id``,
    then the thread portion of ``x-codex-window-id`` (``<thread>:<window>``).
    Parent and spawned reviewer threads share ``session_id`` but have distinct
    ``thread_id`` values — callers that must not cross subagent boundaries
    should use this, not ``session_key_from_headers``.
    """
    for name in ("thread-id", "Thread-Id"):
        raw = headers.get(name)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    meta_raw = headers.get("x-codex-turn-metadata") or headers.get("X-Codex-Turn-Metadata")
    if isinstance(meta_raw, str) and meta_raw.strip():
        try:
            meta = json.loads(meta_raw)
        except json.JSONDecodeError:
            meta = None
        if isinstance(meta, dict):
            thread_id = meta.get("thread_id")
            if isinstance(thread_id, str) and thread_id.strip():
                return thread_id.strip()
    window = headers.get("x-codex-window-id") or headers.get("X-Codex-Window-Id")
    if isinstance(window, str) and window.strip():
        return window.strip().split(":", 1)[0]
    return None


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


def parse_cache_byte_limit(raw: str) -> int | None:
    value = raw.strip()
    if not value:
        return None
    lower = value.lower()
    for suffix, multiplier in _BYTE_SUFFIX_MULTIPLIERS.items():
        if lower.endswith(suffix):
            return int(float(lower[:-1]) * multiplier)
    return int(value)


def max_cache_bytes() -> int:
    raw = os.environ.get("CODEX_SHIM_CHATGPT_CACHE_MAX_BYTES", "").strip()
    if not raw:
        return _DEFAULT_MAX_CACHE_BYTES
    parsed = parse_cache_byte_limit(raw)
    return parsed if parsed is not None and parsed > 0 else _DEFAULT_MAX_CACHE_BYTES


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
        self._write_disk_entry(session_key, response_id, items)

    def _write_disk_entry(self, session_key: str, response_id: str, items: list[Any]) -> None:
        path = self._entry_path(session_key, response_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = {"version": _CACHE_VERSION, "items": copy.deepcopy(items)}
        data = json.dumps(payload, separators=(",", ":"))
        tmp.write_text(data)
        os.replace(tmp, path)
        self._evict_if_needed(session_key)
        self._evict_global_by_size_if_needed()

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
        limit = max_cache_bytes()
        return {
            "session_dirs": session_dirs,
            "file_count": file_count,
            "total_bytes": total_bytes,
            "max_bytes": limit,
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
            self._evict_read_cache_for_path(path)

    def _list_disk_entries(self) -> list[tuple[Path, float, int]]:
        entries: list[tuple[Path, float, int]] = []
        if not self.root.is_dir():
            return entries
        for session_dir in self.root.iterdir():
            if not session_dir.is_dir():
                continue
            for path in session_dir.glob("*.json"):
                if not path.is_file():
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                entries.append((path, stat.st_mtime, stat.st_size))
        return entries

    def _evict_global_by_size_if_needed(self) -> None:
        limit = max_cache_bytes()
        entries = self._list_disk_entries()
        total_bytes = sum(size for _, _, size in entries)
        if total_bytes <= limit:
            return
        entries.sort(key=lambda entry: entry[1])
        for path, _, size in entries:
            if total_bytes <= limit:
                break
            try:
                path.unlink()
            except OSError:
                continue
            self._evict_read_cache_for_path(path)
            total_bytes -= size

    def _evict_read_cache_for_path(self, path: Path) -> None:
        resolved = path.resolve()
        stale: list[tuple[str, str]] = []
        for session_key, response_id in self._read_cache:
            if self._entry_path(session_key, response_id).resolve() == resolved:
                stale.append((session_key, response_id))
        for key in stale:
            self._read_cache.pop(key, None)

    @staticmethod
    def _response_id_from_filename(name: str) -> str:
        if name.endswith(".json"):
            return name[: -len(".json")]
        return name
