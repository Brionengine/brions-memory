#!/usr/bin/env python3
"""
Permanently erase memories, including every archived version. For Brion, not for tools.

forget only archives (see sql/002_memory_archive.sql), so it can be undone. That
leaves no way to truly erase something that must not be kept -- a credential
pasted into a memory, say. This is that way, and it is deliberately NOT an MCP
tool: the archive exists because an LLM reading untrusted text can call tools,
and a purge tool would hand back the exact capability the archive took away.

    cd "/mnt/c/Brion's Memory"; set -a; . ./.env; set +a
    ~/.local/share/brions-memory/venv/bin/python -m brions_memory.purge mem_semantic_ab12cd34 [...]
    ... --search "ORIGIN_QC_API_KEY"     find memories (live or archived) containing text

It shows exactly what will be erased and asks for the word "erase" before acting.
"""

from __future__ import annotations

import argparse
import os
import sys

import psycopg
from psycopg.rows import DictRow, dict_row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("memory_ids", nargs="*")
    ap.add_argument("--search", help="list memory ids whose live or archived text contains this")
    args = ap.parse_args()

    dsn = os.environ.get("BRIONS_MEMORY_DB_URL")
    if not dsn:
        print("BRIONS_MEMORY_DB_URL is not set; source .env first.", file=sys.stderr)
        return 2

    with psycopg.Connection[DictRow].connect(dsn, row_factory=dict_row) as conn:
        if args.search:
            rows = conn.execute(
                """SELECT memory_id, 'live' AS where_, left(content_text, 140) AS text
                     FROM memory_nodes WHERE content_text ILIKE %(q)s
                   UNION ALL
                   SELECT memory_id, 'archived', left(content_text, 140)
                     FROM memory_archive WHERE content_text ILIKE %(q)s
                 ORDER BY memory_id""",
                {"q": f"%{args.search}%"},
            ).fetchall()
            for r in rows:
                print(f"{r['memory_id']}  ({r['where_']})  {r['text']!r}")
            print(f"{len(rows)} match(es).")
            return 0

        if not args.memory_ids:
            ap.print_help()
            return 2

        live = conn.execute(
            "SELECT memory_id, left(content_text, 140) AS text FROM memory_nodes WHERE memory_id = ANY(%s)",
            (args.memory_ids,),
        ).fetchall()
        archived = conn.execute(
            "SELECT count(*) AS n FROM memory_archive WHERE memory_id = ANY(%s)", (args.memory_ids,),
        ).fetchone() or {"n": 0}
        if not live and not archived["n"]:
            print("Nothing found for those ids.")
            return 1
        for r in live:
            print(f"live      {r['memory_id']}  {r['text']!r}")
        print(f"archived  {archived['n']} version(s)")

        if input('This cannot be undone. Type "erase" to continue: ').strip() != "erase":
            print("Cancelled; nothing erased.")
            return 1
        n_live = conn.execute("DELETE FROM memory_nodes WHERE memory_id = ANY(%s)", (args.memory_ids,)).rowcount
        n_arch = conn.execute("DELETE FROM memory_archive WHERE memory_id = ANY(%s)", (args.memory_ids,)).rowcount
        print(f"Erased {n_live} live memor{'y' if n_live == 1 else 'ies'} and {n_arch} archived version(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
