from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from codex_shim import chatgpt_conversation_cache as cache_mod
from codex_shim.chatgpt_conversation_cache import (
    ChatgptConversationCache,
    max_cache_bytes,
    parse_cache_byte_limit,
    sanitize_path_segment,
    sanitize_response_filename,
    session_key_from_headers,
    thread_id_from_headers,
)


def test_session_key_prefers_session_id():
    assert session_key_from_headers({"session-id": "sess-1", "thread-id": "thread-2"}) == "sess-1"


def test_session_key_falls_back_to_thread_id():
    assert session_key_from_headers({"thread-id": "thread-2"}) == "thread-2"


def test_session_key_unscoped_when_missing(capsys, monkeypatch):
    monkeypatch.setattr(cache_mod, "_unscoped_warned", False)
    assert session_key_from_headers({}) == "_unscoped"
    assert session_key_from_headers({}) == "_unscoped"
    captured = capsys.readouterr()
    assert captured.out.count("_unscoped partition") == 1


def test_thread_id_from_turn_metadata_and_window():
    assert thread_id_from_headers({"thread-id": "thread-explicit"}) == "thread-explicit"
    assert (
        thread_id_from_headers(
            {"x-codex-turn-metadata": json.dumps({"thread_id": "thread-meta", "session_id": "sess"})}
        )
        == "thread-meta"
    )
    assert thread_id_from_headers({"x-codex-window-id": "thread-window:13"}) == "thread-window"
    assert thread_id_from_headers({}) is None


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


def test_second_terminal_put_overwrites_disk(tmp_path: Path):
    cache = ChatgptConversationCache(tmp_path)
    first = [{"type": "message", "role": "user", "content": "first"}]
    second = [{"type": "message", "role": "user", "content": "second"}]
    cache.put("sess-1", "resp_1", first, terminal=True)
    cache.put("sess-1", "resp_1", second, terminal=True)
    path = tmp_path / sanitize_path_segment("sess-1") / sanitize_response_filename("resp_1")
    assert json.loads(path.read_text())["items"] == second
    assert cache.get("sess-1", "resp_1") == second


def test_session_isolation(tmp_path: Path):
    cache = ChatgptConversationCache(tmp_path)
    cache.put("sess-a", "resp_shared", [{"content": "a"}], terminal=True)
    cache.put("sess-b", "resp_shared", [{"content": "b"}], terminal=True)
    assert cache.get("sess-a", "resp_shared") == [{"content": "a"}]
    assert cache.get("sess-b", "resp_shared") == [{"content": "b"}]


def test_latest_returns_most_recent_put_for_session(tmp_path: Path):
    cache = ChatgptConversationCache(tmp_path)
    cache.put("sess-1", "resp_1", [{"n": 1}], terminal=True)
    cache.put("sess-1", "resp_2", [{"n": 2}], terminal=True)
    cache.put("sess-other", "resp_9", [{"n": 9}], terminal=True)
    assert cache.latest("sess-1") == [{"n": 2}]
    assert cache.latest("missing") is None


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


def test_parse_cache_byte_limit_accepts_suffixes():
    assert parse_cache_byte_limit("512M") == 512 * 1024 * 1024
    assert parse_cache_byte_limit("1g") == 1024**3


def test_global_size_eviction_removes_oldest_across_sessions(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CODEX_SHIM_CHATGPT_CACHE_MAX_BYTES", "200")
    cache = ChatgptConversationCache(tmp_path)
    cache.put("sess-a", "resp_a", [{"content": "a" * 80}], terminal=True)
    time.sleep(0.01)
    cache.put("sess-b", "resp_b", [{"content": "b" * 80}], terminal=True)
    time.sleep(0.01)
    cache.put("sess-c", "resp_c", [{"content": "c" * 80}], terminal=True)
    remaining = list(tmp_path.rglob("*.json"))
    total = sum(path.stat().st_size for path in remaining)
    assert total <= max_cache_bytes()
    assert cache.get("sess-a", "resp_a") is None
    assert cache.get("sess-c", "resp_c") is not None


def test_concurrent_puts_over_size_limit_do_not_race(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CODEX_SHIM_CHATGPT_CACHE_MAX_BYTES", "4000")
    cache = ChatgptConversationCache(tmp_path)

    def _write(index: int) -> None:
        cache.put(
            f"sess-{index % 8}",
            f"resp_{index}",
            [{"content": f"{index}-{'x' * 40}"}],
            terminal=True,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_write, range(80)))

    remaining = list(tmp_path.rglob("*.json"))
    total = sum(path.stat().st_size for path in remaining)
    assert total <= max_cache_bytes()
    cache.stats()


def test_memory_lru_caps_resident_entries(tmp_path: Path):
    cache = ChatgptConversationCache(tmp_path, max_memory_entries=2)
    cache.put("sess", "resp_a", [{"n": 1}], terminal=True)
    cache.put("sess", "resp_b", [{"n": 2}], terminal=True)
    cache.get("sess", "resp_a")
    cache.put("sess", "resp_c", [{"n": 3}], terminal=True)
    assert cache.stats()["read_cache_entries"] == 2
    assert cache.get("sess", "resp_b") == [{"n": 2}]
    assert cache.get("sess", "resp_a") == [{"n": 1}]
    assert cache.get("sess", "resp_c") == [{"n": 3}]


def test_disk_lru_keeps_recently_read_entry(tmp_path: Path, monkeypatch):
    cache = ChatgptConversationCache(tmp_path)
    payload = [{"content": "x" * 80}]
    cache.put("sess-a", "resp_a", payload, terminal=True)
    size_a = (
        tmp_path / sanitize_path_segment("sess-a") / sanitize_response_filename("resp_a")
    ).stat().st_size
    monkeypatch.setenv("CODEX_SHIM_CHATGPT_CACHE_MAX_BYTES", str(size_a * 2 + 32))
    cache.put("sess-b", "resp_b", payload, terminal=True)
    assert cache.get("sess-a", "resp_a") is not None
    assert cache.get("sess-b", "resp_b") is not None
    cache.get("sess-a", "resp_a")
    cache.put("sess-c", "resp_c", payload, terminal=True)
    remaining = list(tmp_path.rglob("*.json"))
    total = sum(path.stat().st_size for path in remaining)
    assert total <= max_cache_bytes()
    assert cache.get("sess-a", "resp_a") is not None
    assert cache.get("sess-b", "resp_b") is None
    assert cache.get("sess-c", "resp_c") is not None


def test_prune_until_keeps_recently_read_entry(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CODEX_SHIM_CHATGPT_CACHE_MAX_BYTES", "10000")
    cache = ChatgptConversationCache(tmp_path)
    payload = [{"content": "x" * 80}]
    cache.put("sess-a", "resp_a", payload, terminal=True)
    cache.put("sess-b", "resp_b", payload, terminal=True)
    cache.put("sess-c", "resp_c", payload, terminal=True)
    cache.get("sess-c", "resp_c")
    size_c = (
        tmp_path / sanitize_path_segment("sess-c") / sanitize_response_filename("resp_c")
    ).stat().st_size
    after = cache.prune_until(size_c + 32)
    assert after["total_bytes"] <= size_c + 32
    assert cache.get("sess-c", "resp_c") is not None


def test_repeated_puts_do_not_rescan_disk_tree(tmp_path: Path, monkeypatch):
    cache = ChatgptConversationCache(tmp_path)
    cache.put("sess", "resp_a", [{"n": 1}], terminal=True)
    scans = {"n": 0}
    original_iterdir = Path.iterdir

    def counting_iterdir(self):
        if self.resolve() == cache.root.resolve():
            scans["n"] += 1
        return original_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", counting_iterdir)
    cache.put("sess", "resp_b", [{"n": 2}], terminal=True)
    cache.put("sess", "resp_c", [{"n": 3}], terminal=True)
    cache.stats()
    assert scans["n"] == 0


@pytest.mark.asyncio
async def test_aput_round_trip_on_event_loop(tmp_path: Path):
    cache = ChatgptConversationCache(tmp_path)
    items = [{"type": "message", "role": "user", "content": "async"}]
    await cache.aput("sess", "resp_1", items, terminal=True)
    assert cache.get("sess", "resp_1") == items
