#!/usr/bin/env python3
"""
Sync Claude Code's file memories into Brion's Memory.

Why: the cloud store stopped learning on 2026-09-13. Everything in it came from a
one-time import of claude-mem, and claude-mem itself stopped writing on 09-12. Meanwhile
Claude Code keeps curated, one-fact-per-file memories under
/root/.claude/projects/*/memory/*.md -- the measured results, standing rules and
corrections that recall most needs. This makes those reach the recall hook.

One file = one memory, keyed by its path. Re-running is safe:
  new file         -> store()
  file changed     -> update() in place (same memory_id, re-embedded, re-entangled)
  file unchanged   -> skipped
Nothing is ever deleted. A file that disappears leaves its memory where it is.

  python -m migrations.sync_file_memory              # dry run: prints the plan
  python -m migrations.sync_file_memory --apply      # writes
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from migrations.import_history import redact  # noqa: E402  same secret patterns as the importer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sync-file-memory")

ROOT = Path("/root/.claude/projects")
SOURCE = "claude-code/file-memory"
# feedback files are guidance on how to work -> procedural; the rest are facts -> semantic
TYPE_MAP = {"feedback": "procedural"}
FRONT = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.S)


def parse(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    m = FRONT.match(text)
    if not m:
        return None
    head, body = m.group(1), m.group(2).strip()
    field = lambda k: (re.search(rf"^\s*{k}:\s*(.+)$", head, re.M) or [None, ""])[1].strip().strip('"')
    name, desc, ftype = field("name") or path.stem, field("description"), field("type") or "project"
    content = redact(f"{name}: {desc}\n\n{body}" if desc else f"{name}\n\n{body}")
    return {
        "key": f"{path.parent.parent.name}/{path.name}",
        "project": path.parent.parent.name,
        "type": TYPE_MAP.get(ftype, "semantic"),
        "file_type": ftype,
        "content": content,
        "sha": hashlib.sha256(content.encode()).hexdigest(),
        "mtime": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc),
    }


def collect():
    for p in sorted(ROOT.glob("*/memory/*.md")):
        if p.name == "MEMORY.md" or "scratchpad" in str(p.parent.parent):
            continue
        rec = parse(p)
        if rec:
            yield rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write; default is a dry run")
    ap.add_argument("--dsn", default=os.environ.get("BRIONS_MEMORY_DB_URL"))
    args = ap.parse_args()
    if not args.dsn:
        logger.error("no DSN; set BRIONS_MEMORY_DB_URL")
        return 2

    import psycopg
    from psycopg.rows import dict_row

    records = list(collect())
    with psycopg.connect(args.dsn, row_factory=dict_row) as conn:
        known = {r["k"]: r for r in conn.execute(
            "SELECT memory_id, metadata->>'key' k, metadata->>'sha' sha FROM memory_nodes "
            "WHERE metadata->>'source' = %s", (SOURCE,))}

    plan = {"new": [], "changed": [], "same": []}
    for rec in records:
        old = known.get(rec["key"])
        plan["same" if old and old["sha"] == rec["sha"] else "changed" if old else "new"].append(rec)
    logger.info("%d files: %d new, %d changed, %d unchanged",
                len(records), len(plan["new"]), len(plan["changed"]), len(plan["same"]))
    for rec in (plan["new"] + plan["changed"])[:8]:
        print(f"  [{rec['type']:<10}] {rec['key'][:70]}")

    if not args.apply:
        print("dry run -- nothing written. Re-run with --apply.")
        return 0

    from brions_memory.encoder import Encoder
    from brions_memory.store import MemoryStore
    store = MemoryStore(dsn=args.dsn, encoder=Encoder(allow_fallback=False))
    meta = lambda rec: {"source": SOURCE, "key": rec["key"], "sha": rec["sha"], "file_type": rec["file_type"]}
    done = 0
    for rec in plan["new"]:
        store.store(rec["content"], memory_type=rec["type"], metadata=meta(rec), project=rec["project"],
                    signature_extra=rec["key"], created=rec["mtime"])
        done += 1
    for rec in plan["changed"]:
        store.update(known[rec["key"]]["memory_id"], content=rec["content"], memory_type=rec["type"],
                     metadata=meta(rec))
        done += 1
    store.close()
    logger.info("wrote %d memories", done)
    return 0


if __name__ == "__main__":
    sys.exit(main())
