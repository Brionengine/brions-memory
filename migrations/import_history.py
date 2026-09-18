#!/usr/bin/env python3
"""
Import Brion's existing memory into the cloud store.

Sources, in the order they are read:
  1. /root/.claude-mem/claude-mem.db   observations, session summaries, prompts
  2. <legacy-root>/claude_memory/*.db       conversations, project context
  3. <legacy-root>/claude_memory_mcp/*.db   memories, long-term memories

Written as a bulk path rather than repeated MemoryStore.store() calls for two
reasons that both showed up in measurement: embeddings are far cheaper in
batches than one at a time, and entangling on every insert is O(n) per write,
so a 6,000-row import would spend most of its time re-querying what it had just
written. Entanglement runs once at the end instead.

Original timestamps are preserved. A memory that says it is from August should
not claim to be from today.

Idempotent: reruns skip anything already present by content signature.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import psycopg
from psycopg.rows import dict_row

from brions_memory.encoder import Encoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s",
                    stream=sys.stderr)
logger = logging.getLogger("import")

BATCH = 128

# Credentials pasted into past prompts must not be copied into a cloud database.
# Measured 2026-09-12: 6 source records matched these patterns.
import re as _re
SECRET_PATTERNS = _re.compile(
    r"(gh[opsu]_[A-Za-z0-9]{20,}|dop_v1_[a-f0-9]{20,}|sk-[A-Za-z0-9_-]{20,}"
    r"|AKIA[0-9A-Z]{16}|AVNS_[A-Za-z0-9_-]{10,}"
    # 2026-09-18: a Google API key (AIza...) in a file memory passed the list above untouched
    r"|AIza[0-9A-Za-z_-]{35}|xox[baprs]-[A-Za-z0-9-]{10,}|glpat-[A-Za-z0-9_-]{20,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----)"
)
DSN_PASSWORD = _re.compile(r"(postgres(?:ql)?://[^\s:/]+:)[^\s@]+@")


def redact(text: str) -> str:
    text = SECRET_PATTERNS.sub("[REDACTED]", text)
    return DSN_PASSWORD.sub(r"\1[REDACTED]@", text)


def _rows(db_path: str, query: str) -> Iterator[sqlite3.Row]:
    if not os.path.exists(db_path):
        logger.warning("missing source, skipping: %s", db_path)
        return
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        for row in conn.execute(query):
            yield row
    except sqlite3.Error as exc:
        logger.warning("%s: %s", db_path, exc)
    finally:
        conn.close()


def _ts(value: Any) -> Optional[datetime]:
    """Parse the several timestamp shapes these databases use."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # claude-mem stores epoch milliseconds
        seconds = value / 1000.0 if value > 1e11 else float(value)
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    text = str(value).strip().replace("Z", "+00:00")
    for parse in (datetime.fromisoformat,):
        try:
            dt = parse(text)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def _json_list(value: Any) -> List[Any]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else [parsed]
    except (json.JSONDecodeError, TypeError):
        return []


# ---------------------------------------------------------------------------
# Source readers -> (content, memory_type, importance, created, project, metadata)
# ---------------------------------------------------------------------------

def from_claude_mem(db: str) -> Iterator[Tuple]:
    """Observations carry the most signal: a title, a claim, and the reasoning."""
    for r in _rows(db, """
            SELECT title, subtitle, narrative, facts, concepts, type, project,
                   created_at, created_at_epoch, files_modified
              FROM observations"""):
        parts = [p for p in (r["title"], r["subtitle"], r["narrative"]) if p]
        facts = _json_list(r["facts"])
        if facts:
            parts.append(" ".join(str(f) for f in facts[:6]))
        content = " — ".join(parts).strip()
        if not content:
            continue

        # An observation that changed files is a thing that happened; one that
        # only concluded something is knowledge.
        touched = _json_list(r["files_modified"])
        mem_type = "episodic" if touched else "semantic"

        yield (content, mem_type, 1.2 if touched else 1.0,
               _ts(r["created_at"]) or _ts(r["created_at_epoch"]),
               r["project"],
               {"source": "claude-mem/observations", "obs_type": r["type"],
                "concepts": _json_list(r["concepts"])})

    for r in _rows(db, """
            SELECT request, investigated, learned, completed, next_steps,
                   project, created_at, created_at_epoch
              FROM session_summaries"""):
        parts = []
        for label, key in (("Asked", "request"), ("Found", "investigated"),
                           ("Learned", "learned"), ("Done", "completed"),
                           ("Next", "next_steps")):
            if r[key]:
                parts.append(f"{label}: {r[key]}")
        if not parts:
            continue
        yield (" | ".join(parts), "episodic", 1.5,
               _ts(r["created_at"]) or _ts(r["created_at_epoch"]),
               r["project"], {"source": "claude-mem/session_summaries"})

    for r in _rows(db, "SELECT * FROM user_prompts"):
        keys = r.keys()
        text = next((r[k] for k in ("text", "prompt", "content") if k in keys and r[k]), None)
        if not text:
            continue
        created = next((r[k] for k in ("created_at", "created_at_epoch") if k in keys), None)
        yield (str(text), "episodic", 1.3, _ts(created),
               r["project"] if "project" in keys else None,
               {"source": "claude-mem/user_prompts", "speaker": "brion"})


def from_claude_memory(root: Path) -> Iterator[Tuple]:
    """The Dec 2025 system: conversations, project context, long-term memories."""
    conv = root / "claude_memory" / "conversations.db"
    for r in _rows(str(conv), """
            SELECT user_message, claude_response, timestamp, importance_score,
                   project_id, context_tags FROM conversations"""):
        if not r["user_message"]:
            continue
        content = f"Brion asked: {r['user_message']}"
        if r["claude_response"]:
            content += f" | Answer: {str(r['claude_response'])[:600]}"
        yield (content, "episodic",
               float(r["importance_score"] or 5) / 5.0,
               _ts(r["timestamp"]), r["project_id"],
               {"source": "claude_memory/conversations",
                "tags": _json_list(r["context_tags"])})

    ctx = root / "claude_memory" / "context.db"
    for r in _rows(str(ctx), """
            SELECT context_key, context_data, context_type, created_at
              FROM context_memory"""):
        if not r["context_key"]:
            continue
        yield (f"{r['context_key']} = {r['context_data']}", "semantic", 1.5,
               _ts(r["created_at"]), None,
               {"source": "claude_memory/context", "kind": r["context_type"]})

    mcp = root / "claude_memory_mcp" / "memory.db"
    for r in _rows(str(mcp), """
            SELECT content, memory_type, importance, tags, project_context, timestamp
              FROM long_term_memory"""):
        if not r["content"]:
            continue
        yield (str(r["content"]), "semantic",
               float(r["importance"] or 5) / 5.0,
               _ts(r["timestamp"]), r["project_context"],
               {"source": "claude_memory_mcp/long_term"})


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claude-mem-db", default="/root/.claude-mem/claude-mem.db")
    ap.add_argument("--legacy-root", default="/mnt/c/Quantum Cyber Bitcoin/Virtual_Quantum_Computer_Extracted/Virtual Quantum Computer")
    ap.add_argument("--dsn", default=os.environ.get("BRIONS_MEMORY_DB_URL"))
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-entangle", action="store_true")
    args = ap.parse_args()

    if not args.dsn:
        logger.error("no DSN; set BRIONS_MEMORY_DB_URL")
        return 2

    encoder = Encoder(allow_fallback=False)
    encoder.embed("warmup")
    logger.info("embedding model loaded")

    records: List[Tuple] = list(from_claude_mem(args.claude_mem_db))
    records += list(from_claude_memory(Path(args.legacy_root)))
    if args.limit:
        records = records[:args.limit]
    logger.info("collected %d source records", len(records))

    if args.dry_run:
        for content, mtype, imp, created, project, meta in records[:5]:
            print(f"[{mtype}] {str(created)[:19]} ({project}) {content[:110]}")
        print(f"... {len(records)} total")
        return 0

    inserted = skipped = 0
    with psycopg.connect(args.dsn, row_factory=dict_row) as conn:
        existing = {
            r["quantum_signature"]
            for r in conn.execute("SELECT quantum_signature FROM memory_nodes")
        }
        logger.info("%d memories already present", len(existing))

        pending: List[Tuple] = []
        redacted = 0
        for content, mtype, imp, created, project, meta in records:
            clean = redact(content)
            if clean != content:
                redacted += 1
                content = clean
            signature = encoder.signature(content, mtype)
            if signature in existing:
                skipped += 1
                continue
            existing.add(signature)
            pending.append((content, mtype, imp, created, project, meta, signature))

            if len(pending) >= BATCH:
                inserted += _flush(conn, encoder, pending)
                pending.clear()
                logger.info("inserted %d / skipped %d", inserted, skipped)

        if pending:
            inserted += _flush(conn, encoder, pending)
        conn.commit()

    logger.info("import complete: %d inserted, %d duplicates skipped, %d redacted",
                inserted, skipped, redacted)

    if not args.skip_entangle and inserted:
        logger.info("building entanglements (this is the slow pass)")
        from brions_memory.store import MemoryStore
        store = MemoryStore(dsn=args.dsn, encoder=encoder)
        built = _entangle_all(store)
        logger.info("created %d entanglement links", built)
        store.close()
    return 0


def _flush(conn, encoder: Encoder, batch: List[Tuple]) -> int:
    """Embed the whole batch in one model call, then insert."""
    texts = [b[0] for b in batch]
    vectors = encoder.model.encode(texts, batch_size=32, show_progress_bar=False)

    rows = []
    for (content, mtype, imp, created, project, meta, signature), vec in zip(batch, vectors):
        vec = np.asarray(vec, dtype=np.float64)
        norm = np.linalg.norm(vec)
        vec = vec / norm if norm > 0 else vec

        amplitude = np.abs(vec)
        phase = np.where(vec < 0, np.pi, 0.0)
        state = amplitude * np.exp(1j * phase)
        state_norm = np.linalg.norm(state)
        if state_norm > 0:
            state = state / state_norm

        rows.append((
            f"mem_{mtype}_{uuid.uuid4().hex[:8]}",
            psycopg.types.json.Jsonb({"value": content}),
            content, mtype, vec.tolist(), encoder.pack_state(state), len(state),
            float(imp), signature, psycopg.types.json.Jsonb(meta or {}),
            created or datetime.now(timezone.utc),
            created or datetime.now(timezone.utc),
            project, "import",
        ))

    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO memory_nodes
                   (memory_id, content, content_text, memory_type, embedding,
                    quantum_state, quantum_dimension, importance, quantum_signature,
                    metadata, creation_time, last_accessed, project, origin_node)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (quantum_signature) DO NOTHING""",
            rows,
        )
    return len(rows)


def _entangle_all(store, min_strength: float = 0.3, neighbours: int = 12) -> int:
    """
    One ANN pass per memory, nearest `neighbours` only.

    The original compared every new memory against every existing one. At this
    corpus size that is roughly 18 million pairs; the ANN index makes it linear.
    """
    created = 0
    with store.pool.connection() as conn:
        ids = [r["memory_id"] for r in conn.execute(
            "SELECT memory_id FROM memory_nodes WHERE embedding IS NOT NULL")]
        logger.info("entangling %d memories", len(ids))

        for n, memory_id in enumerate(ids, 1):
            created += store.entangle_new(memory_id, candidate_limit=neighbours)
            if n % 500 == 0:
                logger.info("  %d / %d  (%d links)", n, len(ids), created)
    return created


if __name__ == "__main__":
    raise SystemExit(main())
