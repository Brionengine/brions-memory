#!/usr/bin/env python3
"""
Sync qmem's extracted observations into Brion's Memory.

Why: claude-mem stopped writing on 2026-09-12, and qmem -- which has captured every turn
since -- was never a source for this store. Its drain had also crashed on every run since
09-01, so on 2026-09-19 the newest session recall could show was a week old. With the drain
fixed, this carries its output across.

Two kinds of record, both idempotent:
  observation -> one memory each, formatted like import_history's claude-mem observations,
                 keyed by qmem's content_hash
  session     -> one episodic summary per session (source "session", which /profile reads
                 as "most recent sessions"), keyed by session_id and updated in place as the
                 session gains observations

Only work after the claude-mem cutoff is carried: qmem's older observations describe the same
turns claude-mem already summarised, and importing both would say everything twice.

  python migrations/sync_qmem.py              # dry run: prints the plan
  python migrations/sync_qmem.py --apply      # writes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from migrations.import_history import _json_list, redact  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sync-qmem")

QMEM_DB = os.environ.get("QMEM_DB", "/root/.qmem/qmem.db")
OBS_SOURCE = "qmem/observations"
SESSION_SOURCE = "session"
SESSION_TITLES = 15
PROMPT_CHARS = 300


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _when(ts: str | None, fallback_epoch: int) -> datetime:
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return datetime.fromtimestamp(fallback_epoch, timezone.utc)


def collect(cutoff: datetime):
    """Observations newer than cutoff, each with the turn it came from."""
    con = sqlite3.connect(f"file:{QMEM_DB}?mode=ro", uri=True, timeout=15.0)
    con.row_factory = sqlite3.Row
    rows = con.execute("""
        SELECT o.*, q.payload FROM observations o
          LEFT JOIN queue q ON q.unit_hash = o.unit_hash
         ORDER BY o.created_epoch""").fetchall()
    con.close()
    for r in rows:
        unit = json.loads(r["payload"]) if r["payload"] else {}
        when = _when(unit.get("timestamp"), r["created_epoch"])
        if when <= cutoff:
            continue
        parts = [p for p in (r["title"], r["subtitle"], r["narrative"]) if p]
        facts = _json_list(r["facts"])
        if facts:
            parts.append(" ".join(str(f) for f in facts[:6]))
        content = redact(" — ".join(parts).strip())
        if not content:
            continue
        yield {
            "key": r["content_hash"] or _sha(content),
            "content": content,
            # same rule as the claude-mem importer: touching files is a thing that happened
            "type": "episodic" if _json_list(r["files"]) else "semantic",
            "obs_type": r["type"],
            "concepts": _json_list(r["concepts"]),
            "title": r["title"],
            "prompt": " ".join(str(unit.get("prompt") or "").split())[:PROMPT_CHARS],
            "session": r["session_id"] or unit.get("session_id") or "",
            "project": r["project"],
            "when": when,
        }


def sessions(observations):
    by = {}
    for o in observations:
        if o["session"]:
            by.setdefault(o["session"], []).append(o)
    for sid, obs in by.items():
        obs.sort(key=lambda o: o["when"])
        first, last = obs[0]["when"], obs[-1]["when"]
        asked = next((o["prompt"] for o in obs if o["prompt"]), "")
        titles = list(dict.fromkeys(o["title"] for o in obs if o["title"]))
        done = "; ".join(titles[:SESSION_TITLES]) + (f"; +{len(titles) - SESSION_TITLES} more"
                                                     if len(titles) > SESSION_TITLES else "")
        span = first.date().isoformat() if first.date() == last.date() else \
            f"{first.date().isoformat()} to {last.date().isoformat()}"
        content = redact(f"Session in {obs[0]['project']} ({span}) | Asked: {asked} | Done: {done}")
        yield {"key": f"qmem-session/{sid}", "session": sid, "project": obs[0]["project"],
               "content": content, "when": last}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write; default is a dry run")
    ap.add_argument("--dsn", default=os.environ.get("BRIONS_MEMORY_DB_URL"))
    args = ap.parse_args()
    if not args.dsn:
        logger.error("no DSN; set BRIONS_MEMORY_DB_URL")
        return 2
    if not os.path.exists(QMEM_DB):
        logger.info("no qmem database at %s; nothing to sync", QMEM_DB)
        return 0

    import psycopg
    from psycopg.rows import DictRow, dict_row

    with psycopg.Connection[DictRow].connect(args.dsn, row_factory=dict_row) as conn:
        cutoff = conn.execute(
            "SELECT max(creation_time) t FROM memory_nodes WHERE metadata->>'source' LIKE 'claude-mem/%%'"
        ).fetchone()["t"] or datetime.min.replace(tzinfo=timezone.utc)
        known = {r["k"]: r for r in conn.execute(
            "SELECT memory_id, metadata->>'key' k, metadata->>'sha' sha FROM memory_nodes "
            "WHERE metadata->>'source' IN (%s, %s)", (OBS_SOURCE, SESSION_SOURCE))}

    observations = list(collect(cutoff))
    new_obs = [o for o in observations if o["key"] not in known]
    sess_new, sess_changed = [], []
    for s in sessions(observations):
        s["sha"] = _sha(s["content"])
        old = known.get(s["key"])
        if not old:
            sess_new.append(s)
        elif old["sha"] != s["sha"]:
            sess_changed.append(s)
    logger.info("cutoff %s: %d observations (%d new); sessions %d new, %d changed",
                cutoff.isoformat(), len(observations), len(new_obs), len(sess_new), len(sess_changed))
    for rec in (sess_new + sess_changed)[:5]:
        print(f"  [session] {rec['content'][:110]}")
    for rec in new_obs[:5]:
        print(f"  [{rec['type']:<9}] {rec['content'][:110]}")

    if not args.apply or not (new_obs or sess_new or sess_changed):
        if not args.apply:
            print("dry run -- nothing written. Re-run with --apply.")
        return 0

    from brions_memory.encoder import Encoder
    from brions_memory.store import MemoryStore
    store = MemoryStore(dsn=args.dsn, encoder=Encoder(allow_fallback=False))
    for o in new_obs:
        store.store(o["content"], memory_type=o["type"], project=o["project"], session_id=o["session"],
                    importance=1.2 if o["type"] == "episodic" else 1.0, signature_extra=o["key"],
                    created=o["when"], metadata={"source": OBS_SOURCE, "key": o["key"],
                                                 "obs_type": o["obs_type"], "concepts": o["concepts"]})
    meta = lambda s: {"source": SESSION_SOURCE, "key": s["key"], "sha": s["sha"]}
    for s in sess_new:
        store.store(s["content"], memory_type="episodic", importance=1.5, project=s["project"],
                    session_id=s["session"], signature_extra=s["key"], created=s["when"], metadata=meta(s))
    for s in sess_changed:
        store.update(known[s["key"]]["memory_id"], content=s["content"], metadata=meta(s))
    store.close()
    logger.info("wrote %d observations, %d new sessions, %d updated sessions",
                len(new_obs), len(sess_new), len(sess_changed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
