from __future__ import annotations

import sqlite3

from codex_shim.thread_migrate import migrate_thread_providers, state_db_paths


def _make_state_db(tmp_path, name: str = "state_5.sqlite") -> sqlite3.Connection:
    db_path = tmp_path / name
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE threads (id TEXT PRIMARY KEY, model_provider TEXT NOT NULL)"
    )
    return conn


def test_state_db_paths_globs_codex_home(tmp_path):
    (tmp_path / "state_5.sqlite").write_bytes(b"")
    (tmp_path / "state_9.sqlite").write_bytes(b"")
    (tmp_path / "other.sqlite").write_bytes(b"")
    assert [p.name for p in state_db_paths(tmp_path)] == ["state_5.sqlite", "state_9.sqlite"]


def test_migrate_thread_providers_updates_legacy_rows(tmp_path):
    conn = _make_state_db(tmp_path)
    conn.executemany(
        "INSERT INTO threads (id, model_provider) VALUES (?, ?)",
        [("a", "codex_shim"), ("b", "openai"), ("c", "codex_shim")],
    )
    conn.commit()
    conn.close()

    result = migrate_thread_providers(tmp_path)
    assert result == {"updated": 2, "databases": {str(tmp_path / "state_5.sqlite"): 2}}

    conn = sqlite3.connect(tmp_path / "state_5.sqlite")
    rows = conn.execute("SELECT model_provider, COUNT(*) FROM threads GROUP BY model_provider").fetchall()
    conn.close()
    assert sorted(rows) == [("openai", 3)]


def test_migrate_thread_providers_dry_run_is_read_only(tmp_path):
    conn = _make_state_db(tmp_path)
    conn.execute("INSERT INTO threads (id, model_provider) VALUES ('a', 'codex_shim')")
    conn.commit()
    conn.close()

    result = migrate_thread_providers(tmp_path, dry_run=True)
    assert result["updated"] == 1

    conn = sqlite3.connect(tmp_path / "state_5.sqlite")
    row = conn.execute("SELECT model_provider FROM threads WHERE id = 'a'").fetchone()
    conn.close()
    assert row == ("codex_shim",)
