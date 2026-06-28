from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from codex_shim.chatgpt_conversation_cache import (
    ChatgptConversationCache,
    sanitize_path_segment,
    sanitize_response_filename,
    session_key_from_headers,
)


def test_session_key_prefers_session_id():
    assert session_key_from_headers({"session-id": "sess-1", "thread-id": "thread-2"}) == "sess-1"


def test_session_key_falls_back_to_thread_id():
    assert session_key_from_headers({"thread-id": "thread-2"}) == "thread-2"


def test_session_key_unscoped_when_missing(capsys):
    assert session_key_from_headers({}) == "_unscoped"
    assert session_key_from_headers({}) == "_unscoped"
    captured = capsys.readouterr()
    assert captured.out.count("_unscoped partition") == 1


def test_sanitize_response_filename():
    assert sanitize_response_filename("resp_abc") == "resp_abc.json"


def test_put_get_round_trip(tmp_path: Path):
    cache = ChatgptConversationCache(tmp_path)
    items = [{"type": "message", "role": "user", "content": "hi"}]
    cache.put("sess-1", "resp_1", items, terminal=True)
    assert cache.get("sess-1", "resp_1") == items


def test_terminal_false_does_not_write_disk(tmp_path: Path):
    cache = ChatgptConversationCache(tmp_path)
    items = [{"type": "message", "role": "user", "content": "partial"}]
    cache.put("sess-1", "resp_1", items, terminal=False)
    session_dir = tmp_path / sanitize_path_segment("sess-1")
    assert not session_dir.exists() or list(session_dir.glob("*.json")) == []
    assert cache.get("sess-1", "resp_1") == items


def test_terminal_true_writes_disk(tmp_path: Path):
    cache = ChatgptConversationCache(tmp_path)
    items = [{"type": "message", "role": "user", "content": "done"}]
    cache.put("sess-1", "resp_1", items, terminal=False)
    cache.put("sess-1", "resp_1", items, terminal=True)
    path = tmp_path / sanitize_path_segment("sess-1") / sanitize_response_filename("resp_1")
    assert path.is_file()
    payload = json.loads(path.read_text())
    assert payload["version"] == 1
    assert payload["items"] == items


def test_second_terminal_put_does_not_overwrite_file(tmp_path: Path):
    cache = ChatgptConversationCache(tmp_path)
    first = [{"type": "message", "role": "user", "content": "first"}]
    second = [{"type": "message", "role": "user", "content": "second"}]
    cache.put("sess-1", "resp_1", first, terminal=True)
    cache.put("sess-1", "resp_1", second, terminal=True)
    path = tmp_path / sanitize_path_segment("sess-1") / sanitize_response_filename("resp_1")
    assert json.loads(path.read_text())["items"] == first
    assert cache.get("sess-1", "resp_1") == second


def test_session_isolation(tmp_path: Path):
    cache = ChatgptConversationCache(tmp_path)
    cache.put("sess-a", "resp_shared", [{"content": "a"}], terminal=True)
    cache.put("sess-b", "resp_shared", [{"content": "b"}], terminal=True)
    assert cache.get("sess-a", "resp_shared") == [{"content": "a"}]
    assert cache.get("sess-b", "resp_shared") == [{"content": "b"}]


def test_eviction_removes_oldest_by_mtime(tmp_path: Path):
    cache = ChatgptConversationCache(tmp_path)
    session_dir = tmp_path / sanitize_path_segment("sess-1")
    session_dir.mkdir(parents=True)
    paths: list[Path] = []
    for index in range(1025):
        response_id = f"resp_{index:04d}"
        path = session_dir / sanitize_response_filename(response_id)
        path.write_text(json.dumps({"version": 1, "items": [{"n": index}]}))
        paths.append(path)
        time.sleep(0.001)
    cache.put("sess-1", "resp_new", [{"content": "new"}], terminal=True)
    remaining = {path.name for path in session_dir.glob("*.json")}
    assert sanitize_response_filename("resp_0000") not in remaining
    assert sanitize_response_filename("resp_new") in remaining
    assert len(remaining) == 1024


def test_restart_survival_reads_disk(tmp_path: Path):
    first = ChatgptConversationCache(tmp_path)
    items = [{"type": "message", "role": "user", "content": "persisted"}]
    first.put("sess-1", "resp_1", items, terminal=True)
    second = ChatgptConversationCache(tmp_path)
    assert second.get("sess-1", "resp_1") == items
    assert second.stats()["read_cache_entries"] == 1


def test_reader_cache_populated_on_disk_read(tmp_path: Path):
    writer = ChatgptConversationCache(tmp_path)
    items = [{"type": "message", "role": "user", "content": "cached"}]
    writer.put("sess-1", "resp_1", items, terminal=True)
    reader = ChatgptConversationCache(tmp_path)
    assert reader.stats()["read_cache_entries"] == 0
    assert reader.get("sess-1", "resp_1") == items
    assert reader.stats()["read_cache_entries"] == 1
    assert reader.get("sess-1", "resp_1") == items
    assert reader.stats()["read_cache_entries"] == 1
