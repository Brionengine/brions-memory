#!/usr/bin/env python3
"""
The memory archive, for Brion. Not reachable by any MCP tool.

forget and update_memory are MCP tools, so an LLM reading untrusted text (a web
page, a recalled memory) can be steered into calling them. They only archive,
so nothing is lost. The ways OUT of the archive -- reading what was forgotten,
putting it back, erasing it for good -- live here instead: an LLM-reachable
restore could resurface exactly what Brion chose to remove, an LLM-reachable
listing would expose forgotten text (a pasted credential, say) to whatever the
model is reading, and an LLM-reachable purge would undo the whole point.

    cd "/mnt/c/Brion's Memory"; set -a; . ./.env; set +a
    PY=~/.local/share/brions-memory/venv/bin/python
    $PY -m brions_memory.archive list [--memory-id ID]   archived versions, newest first
    $PY -m brions_memory.archive show ARCHIVE_ID          full text of one archived version
    $PY -m brions_memory.archive restore ARCHIVE_ID       put it back (current version archived first)
    $PY -m brions_memory.archive search TEXT              memory ids whose live or archived text matches
    $PY -m brions_memory.archive purge ID [ID ...]        erase forever, every version; asks for "erase"
"""

from __future__ import annotations

import argparse
import os
import sys

import psycopg
from psycopg.rows import DictRow, dict_row


def _connect(dsn: str) -> psycopg.Connection[DictRow]:
    return psycopg.Connection[DictRow].connect(dsn, row_factory=dict_row)


def cmd_list(conn: psycopg.Connection[DictRow], memory_id: str | None, limit: int) -> int:
    rows = conn.execute(
        """SELECT archive_id, memory_id, archive_reason, archived_at, memory_type,
                  left(content_text, 120) AS text
             FROM memory_archive
            WHERE (%(mid)s::text IS NULL OR memory_id = %(mid)s)
         ORDER BY archived_at DESC LIMIT %(limit)s""",
        {"mid": memory_id, "limit": limit},
    ).fetchall()
    for r in rows:
        print(f"{r['archive_id']:>6}  {r['archived_at']:%Y-%m-%d %H:%M}  {r['archive_reason']:<7}  "
              f"{r['memory_id']}  {r['text']!r}")
    print(f"{len(rows)} archived version(s).")
    return 0


def cmd_show(conn: psycopg.Connection[DictRow], archive_id: int) -> int:
    row = conn.execute(
        """SELECT archive_id, memory_id, archive_reason, archived_at, memory_type, project, content_text
             FROM memory_archive WHERE archive_id = %s""",
        (archive_id,),
    ).fetchone()
    if row is None:
        print(f"No archive entry {archive_id}.")
        return 1
    print(f"archive {row['archive_id']}  {row['memory_id']}  {row['archive_reason']} "
          f"{row['archived_at']:%Y-%m-%d %H:%M}  {row['memory_type']}  {row['project'] or ''}\n")
    print(row["content_text"])
    return 0


def cmd_restore(archive_id: int) -> int:
    # The store re-entangles the restored memory, which needs the encoder; import lazily.
    from .store import MemoryStore
    store = MemoryStore()
    try:
        memory_id = store.restore(archive_id)
    except KeyError as exc:
        print(exc)
        return 1
    finally:
        store.close()
    print(f"Restored {memory_id} from archive entry {archive_id}.")
    return 0


def cmd_search(conn: psycopg.Connection[DictRow], text: str) -> int:
    rows = conn.execute(
        """SELECT memory_id, 'live' AS where_, left(content_text, 140) AS text
             FROM memory_nodes WHERE content_text ILIKE %(q)s
           UNION ALL
           SELECT memory_id, 'archived', left(content_text, 140)
             FROM memory_archive WHERE content_text ILIKE %(q)s
         ORDER BY memory_id""",
        {"q": f"%{text}%"},
    ).fetchall()
    for r in rows:
        print(f"{r['memory_id']}  ({r['where_']})  {r['text']!r}")
    print(f"{len(rows)} match(es).")
    return 0


def cmd_purge(conn: psycopg.Connection[DictRow], memory_ids: list[str]) -> int:
    live = conn.execute(
        "SELECT memory_id, left(content_text, 140) AS text FROM memory_nodes WHERE memory_id = ANY(%s)",
        (memory_ids,),
    ).fetchall()
    archived = (conn.execute(
        "SELECT count(*) AS n FROM memory_archive WHERE memory_id = ANY(%s)", (memory_ids,),
    ).fetchone() or {"n": 0})["n"]
    if not live and not archived:
        print("Nothing found for those ids.")
        return 1
    for r in live:
        print(f"live      {r['memory_id']}  {r['text']!r}")
    print(f"archived  {archived} version(s)")
    if input('This cannot be undone. Type "erase" to continue: ').strip() != "erase":
        print("Cancelled; nothing erased.")
        return 1
    n_live = conn.execute("DELETE FROM memory_nodes WHERE memory_id = ANY(%s)", (memory_ids,)).rowcount
    n_arch = conn.execute("DELETE FROM memory_archive WHERE memory_id = ANY(%s)", (memory_ids,)).rowcount
    print(f"Erased {n_live} live memor{'y' if n_live == 1 else 'ies'} and {n_arch} archived version(s).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--memory-id")
    p.add_argument("--limit", type=int, default=30)
    sub.add_parser("show").add_argument("archive_id", type=int)
    sub.add_parser("restore").add_argument("archive_id", type=int)
    sub.add_parser("search").add_argument("text")
    sub.add_parser("purge").add_argument("memory_ids", nargs="+")
    args = ap.parse_args()

    dsn = os.environ.get("BRIONS_MEMORY_DB_URL")
    if not dsn:
        print("BRIONS_MEMORY_DB_URL is not set; source .env first.", file=sys.stderr)
        return 2
    if args.cmd == "restore":
        return cmd_restore(args.archive_id)
    with _connect(dsn) as conn:
        if args.cmd == "list":
            return cmd_list(conn, args.memory_id, args.limit)
        if args.cmd == "show":
            return cmd_show(conn, args.archive_id)
        if args.cmd == "search":
            return cmd_search(conn, args.text)
        return cmd_purge(conn, args.memory_ids)


if __name__ == "__main__":
    sys.exit(main())
