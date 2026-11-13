"""Deny coding agents on this machine any read of the match data bundle.

The committed .claude/settings.json only denies ./data/**. When the bundle lives elsewhere
(MTAH_DATA_DIR), this adds the same protection for that path to .claude/settings.local.json,
which is per-machine and never committed. Idempotent: running it twice changes nothing.

    python3 training/gx10/deny_data.py /srv/mtah2027
"""

import json
import os
import sys

READERS = ("cat", "head", "tail", "less", "more", "strings", "xxd", "od", "grep", "sed", "awk")


def rules_for(path):
    path = os.path.realpath(path).rstrip("/")
    # "//abs/path" is how Claude Code permission rules spell an absolute filesystem path.
    rules = ["Read(/{}/**)".format(path), "Edit(/{}/**)".format(path)]
    rules += ["Bash({} {}*)".format(cmd, path) for cmd in READERS]
    return rules


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    settings = os.path.join(repo, ".claude", "settings.local.json")
    data = {}
    if os.path.isfile(settings):
        with open(settings, encoding="utf-8") as f:
            data = json.load(f)
    deny = data.setdefault("permissions", {}).setdefault("deny", [])
    added = [r for r in rules_for(sys.argv[1]) if r not in deny]
    deny.extend(added)
    os.makedirs(os.path.dirname(settings), exist_ok=True)
    with open(settings, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print("{}: {} deny rules added, {} total".format(settings, len(added), len(deny)))


if __name__ == "__main__":
    main()
