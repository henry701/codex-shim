from __future__ import annotations

from pathlib import Path

import codex_shim.compaction as compaction


def test_compaction_public_api_is_the_package_not_a_shadowed_module():
    assert compaction.__file__ is not None
    assert compaction.__file__.endswith("compaction/__init__.py")
    shadow = Path(compaction.__file__).resolve().parent.parent / "compaction.py"
    assert not shadow.exists(), f"shadowed dead module {shadow} hides the package"
