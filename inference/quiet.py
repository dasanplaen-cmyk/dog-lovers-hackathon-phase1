"""What scripts may print about the provided game while the team's stop rule is on.

Until the organisers answer (board, 2026-11-04): nothing computed from BAR-ATH reaches any AI
service, coding agents included, beyond pass/fail and file hashes. Every script that writes a
BAR-ATH file prints `quiet_report` instead of its usual aggregates; external games keep their full
output. The split of external games (held out, compare-only, dropped) lives in eval/external.py.
"""

from __future__ import annotations

import hashlib
import json

TEST_GAME = "BAR-ATH"


def is_test_game(name: str) -> bool:
    return TEST_GAME in str(name).upper()


def quiet_report(path: str, ok: bool = True) -> str:
    """Path, sha256 and pass/fail of a file written for the provided game. Nothing else."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return json.dumps({"game": TEST_GAME, "out": path, "sha256": h.hexdigest(),
                       "result": "OK" if ok else "FAILED"}, indent=2)
