#!/usr/bin/env python3
"""
Brion's Memory — keep the cloud store learning. Runs as a Claude Code Stop hook.

Measured 2026-09-18: the store had learned nothing since 09-13, because the only writer
was a one-time import. This closes the loop: whenever a Claude Code memory file is newer
than the last sync, a background sync (migrations/sync_file_memory.py --apply) is started.

The hook itself only stats files and maybe forks, so it costs milliseconds; the sync runs
detached, under flock, so overlapping Stops never run two syncs. The marker records when
the last successful sync STARTED, so a file edited mid-sync is picked up next time.

Contract: any failure exits 0 silently. Memory must never be the reason a session stalls.
"""
import glob
import os
import subprocess
import sys

HOME = "/root/.local/share/brions-memory"
MARKER = os.path.join(HOME, "last_sync")
LOCK = os.path.join(HOME, "sync.lock")
LOG = os.path.join(HOME, "sync.log")
PY = os.path.join(HOME, "venv", "bin", "python")
REPO = "/mnt/c/Brion's Memory"
FILES = "/root/.claude/projects/*/memory/*.md"


def newest_change():
    return max((os.path.getmtime(p) for p in glob.glob(FILES)), default=0.0)


def last_sync():
    try:
        with open(MARKER) as fh:
            return float(fh.read().strip() or 0)
    except (OSError, ValueError):
        return 0.0


def main():
    if newest_change() <= last_sync():
        return
    # start time is taken BEFORE the sync reads files; it becomes the marker on success
    script = (
        'start=$(date +%s.%N); cd "$REPO" && set -a && . ./.env && set +a && '
        '"$PY" migrations/sync_file_memory.py --apply && echo "$start" > "$MARKER"'
    )
    env = {**os.environ, "REPO": REPO, "PY": PY, "MARKER": MARKER}
    with open(LOG, "a") as log:
        subprocess.Popen(["flock", "-n", LOCK, "bash", "-c", script], cwd=REPO, env=env,
                         stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
    sys.exit(0)
