from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

_MAX_PATH_SEGMENT_LEN = 200
_CACHE_VERSION = 1
_MAX_CACHED_RESPONSES_PER_SESSION = 1024
_DEFAULT_MAX_CACHE_BYTES = 512 * 1024 * 1024
_DEFAULT_MAX_MEMORY_ENTRIES = 256
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


def configured_max_memory_entries() -> int:
    raw = os.environ.get("CODEX_SHIM_CHATGPT_CACHE_MAX_MEMORY_ENTRIES", "").strip()
    if not raw:
        return _DEFAULT_MAX_MEMORY_ENTRIES
    try:
        parsed = int(raw)
    except ValueError:
        return _DEFAULT_MAX_MEMORY_ENTRIES
    return parsed if parsed > 0 else _DEFAULT_MAX_MEMORY_ENTRIES


class ChatgptConversationCache:
    """Session conversation snapshots: RAM LRU + disk LRU with an incremental index.

    Callers see ``get`` / ``put`` / ``aput``. Disk I/O stays on the caller thread;
    the event loop is fine once eviction is O(1) instead of a full tree walk.
    """

    def __init__(self, root: Path, *, max_memory_entries: int | None = None) -> None:
        self.root = root.expanduser()
        self._max_memory_entries = (
            max_memory_entries if max_memory_entries is not None else configured_max_memory_entries()
        )
        self._memory: OrderedDict[tuple[str, str], list[Any]] = OrderedDict()
        self._disk: OrderedDict[tuple[str, str], int] = OrderedDict()
        self._disk_bytes = 0
        self._session_counts: dict[str, int] = {}
        self._index_loaded = False
        self._lock = threading.RLock()

    def get(self, session_key: str, response_id: str) -> list[Any] | None:
        if not response_id:
            return None
        key = (session_key, response_id)
        with self._lock:
            cached = self._memory.get(key)
            if cached is not None:
                self._touch_locked(key)
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
        with self._lock:
            self._ensure_index_locked()
            if key not in self._disk:
                try:
                    self._note_disk_locked(key, path.stat().st_size)
                except OSError:
                    pass
            self._remember_locked(key, items)
            self._touch_locked(key)
            return copy.deepcopy(items)

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
        with self._lock:
            self._remember_locked(key, items)
            if not terminal:
                return
            self._ensure_index_locked()
        size = self._atomic_write(self._entry_path(session_key, response_id), items)
        with self._lock:
            self._note_disk_locked(key, size)
            self._evict_locked()

    def latest(self, session_key: str) -> list[Any] | None:
        """Most recently stored snapshot for this session, or None."""
        if not session_key:
            return None
        with self._lock:
            self._ensure_index_locked()
            response_id = None
            for key in reversed(self._disk):
                if key[0] == session_key:
                    response_id = key[1]
                    break
            if response_id is None:
                for key in reversed(self._memory):
                    if key[0] == session_key:
                        return copy.deepcopy(self._memory[key])
                return None
        return self.get(session_key, response_id)

    def prune_until(self, max_bytes: int) -> dict[str, int]:
        limit = max(0, int(max_bytes))
        with self._lock:
            self._ensure_index_locked()
            while self._disk_bytes > limit and self._disk:
                victim = next(iter(self._disk))
                self._drop_disk_locked(victim)
            return {
                "session_dirs": len(self._session_counts),
                "file_count": len(self._disk),
                "total_bytes": self._disk_bytes,
                "max_bytes": max_cache_bytes(),
                "read_cache_entries": len(self._memory),
            }

    async def aput(
        self,
        session_key: str,
        response_id: str,
        items: list[Any],
        *,
        terminal: bool = True,
    ) -> None:
        self.put(session_key, response_id, items, terminal=terminal)

    def stats(self) -> dict[str, int]:
        with self._lock:
            self._ensure_index_locked()
            return {
                "session_dirs": len(self._session_counts),
                "file_count": len(self._disk),
                "total_bytes": self._disk_bytes,
                "max_bytes": max_cache_bytes(),
                "read_cache_entries": len(self._memory),
            }

    def _entry_path(self, session_key: str, response_id: str) -> Path:
        session_dir = sanitize_path_segment(session_key)
        return self.root / session_dir / sanitize_response_filename(response_id)

    def _remember_locked(self, key: tuple[str, str], items: list[Any]) -> None:
        self._memory[key] = copy.deepcopy(items)
        self._memory.move_to_end(key)
        while len(self._memory) > self._max_memory_entries:
            self._memory.popitem(last=False)

    def _touch_locked(self, key: tuple[str, str]) -> None:
        if key in self._memory:
            self._memory.move_to_end(key)
        if key in self._disk:
            self._disk.move_to_end(key)

    def _ensure_index_locked(self) -> None:
        if self._index_loaded:
            return
        self._disk.clear()
        self._disk_bytes = 0
        self._session_counts.clear()
        if self.root.is_dir():
            entries: list[tuple[float, tuple[str, str], int]] = []
            for session_dir in self.root.iterdir():
                if not session_dir.is_dir():
                    continue
                session_key = session_dir.name
                for path in session_dir.glob("*.json"):
                    if not path.is_file():
                        continue
                    try:
                        stat = path.stat()
                    except OSError:
                        continue
                    response_id = self._response_id_from_filename(path.name)
                    entries.append((stat.st_mtime, (session_key, response_id), stat.st_size))
            entries.sort(key=lambda item: item[0])
            for _, key, size in entries:
                self._disk[key] = size
                self._disk_bytes += size
                self._session_counts[key[0]] = self._session_counts.get(key[0], 0) + 1
        self._index_loaded = True

    def _atomic_write(self, path: Path, items: list[Any]) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = {"version": _CACHE_VERSION, "items": copy.deepcopy(items)}
        data = json.dumps(payload, separators=(",", ":"))
        tmp.write_text(data)
        os.replace(tmp, path)
        return path.stat().st_size

    def _note_disk_locked(self, key: tuple[str, str], size: int) -> None:
        old = self._disk.pop(key, None)
        if old is not None:
            self._disk_bytes -= old
        else:
            self._session_counts[key[0]] = self._session_counts.get(key[0], 0) + 1
        self._disk[key] = size
        self._disk.move_to_end(key)
        self._disk_bytes += size

    def _evict_locked(self) -> None:
        for session_key, count in list(self._session_counts.items()):
            while count > _MAX_CACHED_RESPONSES_PER_SESSION:
                victim = next((key for key in self._disk if key[0] == session_key), None)
                if victim is None:
                    break
                self._drop_disk_locked(victim)
                count = self._session_counts.get(session_key, 0)
        limit = max_cache_bytes()
        while self._disk_bytes > limit and self._disk:
            victim = next(iter(self._disk))
            self._drop_disk_locked(victim)

    def _drop_disk_locked(self, key: tuple[str, str]) -> None:
        size = self._disk.pop(key, 0)
        self._disk_bytes = max(0, self._disk_bytes - size)
        self._memory.pop(key, None)
        remaining = self._session_counts.get(key[0], 0) - 1
        if remaining <= 0:
            self._session_counts.pop(key[0], None)
        else:
            self._session_counts[key[0]] = remaining
        try:
            self._entry_path(*key).unlink()
        except OSError:
            pass

    @staticmethod
    def _response_id_from_filename(name: str) -> str:
        if name.endswith(".json"):
            return name[: -len(".json")]
        return name
