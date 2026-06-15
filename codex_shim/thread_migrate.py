from __future__ import annotations

import sqlite3
from pathlib import Path

from .settings import OPENAI_PROVIDER_ID, PROVIDER_NAME

DEFAULT_CODEX_HOME = Path.home() / ".codex"


def state_db_paths(codex_home: Path | None = None) -> list[Path]:
    root = codex_home or DEFAULT_CODEX_HOME
    if not root.is_dir():
        return []
    return sorted(root.glob("state_*.sqlite"))


def migrate_thread_providers(
    codex_home: Path | None = None,
    *,
    dry_run: bool = False,
    from_provider: str = PROVIDER_NAME,
    to_provider: str = OPENAI_PROVIDER_ID,
) -> dict[str, int | dict[str, int]]:
    per_db: dict[str, int] = {}
    total = 0
    for db_path in state_db_paths(codex_home):
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM threads WHERE model_provider = ?",
                (from_provider,),
            ).fetchone()
            count = int(row[0]) if row else 0
            if count:
                per_db[str(db_path)] = count
                total += count
                if not dry_run:
                    conn.execute(
                        "UPDATE threads SET model_provider = ? WHERE model_provider = ?",
                        (to_provider, from_provider),
                    )
                    conn.commit()
        finally:
            conn.close()
    return {"updated": total, "databases": per_db}
