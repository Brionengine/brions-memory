"""
Postgres-backed memory store for Brion's Memory.

This is the cloud replacement for the in-process dicts in
quantum_entanglement_memory.py. There, `self.memory_nodes` and
`self.entanglement_clusters` were Python dicts local to one process: nothing
survived a restart and two servers running it would have had separate,
diverging memories. Here the same structures live in one managed Postgres, so
every worker reads and writes the same memory.

Retrieval is two-stage by necessity, not preference:
  1. Postgres narrows by embedding (exact cosine scan -- see _nearest)
  2. the worker computes exact quantum fidelity on those candidates

pgvector is real-valued and cannot hold a complex amplitude, and entanglement
needs the phase. Stage 1 is a recall filter; stage 2 is the actual measure.
"""

from __future__ import annotations

import logging
import os
import socket
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from .encoder import Encoder

logger = logging.getLogger(__name__)

MEMORY_TYPES = (
    "semantic", "episodic", "procedural", "associative",
    "emotional", "contextual", "meta", "quantum",
)

# Thresholds carried over from EntanglementStrength in the original.
WEAK, MODERATE, STRONG, MAXIMAL = 0.3, 0.6, 0.8, 1.0


@dataclass
class Recall:
    memory_id: str
    content: Any
    memory_type: str
    similarity: float      # cosine from pgvector
    fidelity: float        # exact |<s1|s2>|^2
    relevance: float       # fidelity * importance weight
    importance: float
    access_count: int
    last_accessed: datetime
    created: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryStore:
    """Shared cloud memory. Safe to run on several workers at once."""

    def __init__(self,
                 dsn: Optional[str] = None,
                 encoder: Optional[Encoder] = None,
                 node_name: Optional[str] = None,
                 min_size: int = 1,
                 max_size: int = 4):
        self.dsn = dsn or os.environ.get("BRIONS_MEMORY_DB_URL")
        if not self.dsn:
            raise RuntimeError(
                "No database URL. Set BRIONS_MEMORY_DB_URL or pass dsn="
            )
        self.encoder = encoder or Encoder()
        self.node_name = node_name or socket.gethostname()
        self.pool = ConnectionPool(self.dsn, min_size=min_size, max_size=max_size,
                                   kwargs={"row_factory": dict_row}, open=True)

    @staticmethod
    def _nearest(conn, sql: str, params: Sequence[Any]) -> List[Dict[str, Any]]:
        """
        Run a nearest-neighbour query as an exact scan, never through HNSW.

        Measured 2026-09-12/13 at 6.4k rows: the HNSW index missed off-topic
        memories entirely (a canary with exact cosine 0.480, rank 1, was absent
        even at ef_search=400) and averaged recall@20 of 0.953, while the exact
        scan matched ground truth 100%. The price is server time: ~39 ms median
        against HNSW's ~2 ms, growing linearly with the store.

        Callers select only memory_id in the scan and join the wide columns
        back for the winners; carrying content and quantum_state through the
        sort cost ~59 ms. Scoped to this one query so primary-key lookups in
        the same transaction keep their index, and pipelined so the three
        statements cost one network round trip.
        """
        with conn.pipeline():
            conn.execute("SET LOCAL enable_indexscan = off")
            cur = conn.execute(sql, params)
            conn.execute("SET LOCAL enable_indexscan = on")
        return cur.fetchall()

    # -- lifecycle ---------------------------------------------------------

    @contextmanager
    def cursor(self):
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                yield cur

    def close(self) -> None:
        self.pool.close()

    def apply_schema(self, sql_path: str) -> None:
        with open(sql_path, encoding="utf-8") as fh:
            sql = fh.read()
        with self.pool.connection() as conn:
            # File-loaded SQL is a plain str; psycopg's execute() is typed for
            # LiteralString or bytes, so send the schema as UTF-8 bytes.
            conn.execute(sql.encode("utf-8"))
        logger.info("Applied schema from %s", sql_path)

    # -- writing -----------------------------------------------------------

    def store(self,
              content: Any,
              memory_type: str = "semantic",
              importance: float = 1.0,
              metadata: Optional[Dict[str, Any]] = None,
              session_id: Optional[str] = None,
              project: Optional[str] = None,
              entangle: bool = True,
              signature_extra: Optional[str] = None,
              created: Optional[datetime] = None) -> str:
        """
        Store one memory and entangle it with what is already there.

        Deduplicates on the content signature: storing the same content twice
        strengthens the existing memory rather than creating a second copy,
        which is what makes repeated exposure count for something.
        """
        if memory_type not in MEMORY_TYPES:
            raise ValueError(f"unknown memory_type {memory_type!r}")

        signature = self.encoder.signature(
            content if signature_extra is None else f"{signature_extra}\x00{content}",
            memory_type,
        )
        text = content if isinstance(content, str) else str(content)

        with self.pool.connection() as conn:
            existing = conn.execute(
                "SELECT memory_id, access_count FROM memory_nodes WHERE quantum_signature = %s",
                (signature,),
            ).fetchone()
            if existing:
                conn.execute(
                    """UPDATE memory_nodes
                          SET access_count = access_count + 1,
                              last_accessed = now(),
                              importance = LEAST(importance + 0.1, 10.0)
                        WHERE memory_id = %s""",
                    (existing["memory_id"],),
                )
                logger.debug("Reinforced existing memory %s", existing["memory_id"])
                return existing["memory_id"]

            memory_id = f"mem_{memory_type}_{uuid.uuid4().hex[:8]}"
            embedding = self.encoder.embed(content)
            state = self.encoder.quantum_state(content, memory_type)

            conn.execute(
                """INSERT INTO memory_nodes
                       (memory_id, content, content_text, memory_type, embedding,
                        quantum_state, quantum_dimension, importance,
                        quantum_signature, metadata, session_id, origin_node, project,
                        creation_time, last_accessed)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                           COALESCE(%s, now()), COALESCE(%s, now()))""",
                (memory_id,
                 Jsonb({"value": content}),
                 text,
                 memory_type,
                 embedding.tolist(),
                 self.encoder.pack_state(state),
                 len(state),
                 importance,
                 signature,
                 Jsonb(metadata or {}),
                 session_id,
                 self.node_name,
                 project,
                 created,
                 created),
            )

        if entangle:
            self.entangle_new(memory_id)
        return memory_id

    def entangle_new(self, memory_id: str, candidate_limit: int = 50) -> int:
        """
        Link a new memory to its nearest existing memories.

        Only the nearest candidates are scored, not every memory: the original
        compared against all nodes on every store, which is O(n) per write and
        stops being viable well before this store is interesting.
        """
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT embedding, quantum_state, content_text FROM memory_nodes WHERE memory_id = %s",
                (memory_id,),
            ).fetchone()
            if row is None:
                return 0

            state = self.encoder.unpack_state(row["quantum_state"])
            candidates = self._nearest(
                conn,
                """SELECT n.memory_id, n.quantum_state, n.content_text,
                          1 - (n.embedding <=> %s::vector) AS cosine
                     FROM (SELECT memory_id FROM memory_nodes
                            WHERE memory_id <> %s AND embedding IS NOT NULL
                         ORDER BY embedding <=> %s::vector
                            LIMIT %s) nearest
                     JOIN memory_nodes n USING (memory_id)
                 ORDER BY cosine DESC""",
                (row["embedding"], memory_id, row["embedding"], candidate_limit),
            )

            created = 0
            for cand in candidates:
                fidelity = self.encoder.fidelity(
                    state, self.encoder.unpack_state(cand["quantum_state"])
                )
                jaccard = _jaccard(row["content_text"], cand["content_text"])
                # sqrt(fidelity) is |cos|. Blending on the cosine scale rather
                # than cos^2 keeps the WEAK/MODERATE/STRONG thresholds meaning
                # what they meant in the original: cos^2 compresses everything
                # toward zero (cosine 0.55 -> 0.30) and measured at zero
                # entanglements across a realistic memory set.
                strength = min(1.0, (float(np.sqrt(fidelity)) + jaccard) / 2)

                if strength < WEAK:
                    continue

                a, b = sorted((memory_id, cand["memory_id"]))
                conn.execute(
                    """INSERT INTO entanglements
                           (memory_a, memory_b, strength, quantum_overlap, semantic_similarity)
                       VALUES (%s, %s, %s, %s, %s)
                       ON CONFLICT (memory_a, memory_b) DO UPDATE
                           SET strength = EXCLUDED.strength,
                               quantum_overlap = EXCLUDED.quantum_overlap,
                               semantic_similarity = EXCLUDED.semantic_similarity,
                               updated_at = now()""",
                    (a, b, strength, fidelity, jaccard),
                )
                created += 1

        logger.debug("Entangled %s with %d memories", memory_id, created)
        return created

    # -- full access: edit and delete -------------------------------------

    def update(self,
               memory_id: str,
               content: Optional[Any] = None,
               importance: Optional[float] = None,
               memory_type: Optional[str] = None,
               metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Edit a memory in place. Changing content re-encodes it and rebuilds its
        entanglements, since its links were computed from the old meaning.
        """
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT content, memory_type, metadata FROM memory_nodes WHERE memory_id = %s",
                (memory_id,),
            ).fetchone()
            if row is None:
                return False

            new_type = memory_type or row["memory_type"]
            if new_type not in MEMORY_TYPES:
                raise ValueError(f"unknown memory_type {new_type!r}")
            merged_meta = {**(row["metadata"] or {}), **(metadata or {})}

            if content is not None:
                text = content if isinstance(content, str) else str(content)
                state = self.encoder.quantum_state(content, new_type)
                conn.execute(
                    """UPDATE memory_nodes
                          SET content = %s, content_text = %s, memory_type = %s,
                              embedding = %s, quantum_state = %s, quantum_dimension = %s,
                              quantum_signature = %s, metadata = %s,
                              importance = COALESCE(%s, importance)
                        WHERE memory_id = %s""",
                    (Jsonb({"value": content}), text, new_type,
                     self.encoder.embed(content).tolist(), self.encoder.pack_state(state),
                     len(state),
                     self.encoder.signature(f"{memory_id}\x00{text}", new_type),
                     Jsonb(merged_meta), importance, memory_id),
                )
                conn.execute(
                    "DELETE FROM entanglements WHERE memory_a = %s OR memory_b = %s",
                    (memory_id, memory_id),
                )
            else:
                conn.execute(
                    """UPDATE memory_nodes
                          SET memory_type = %s, metadata = %s,
                              importance = COALESCE(%s, importance)
                        WHERE memory_id = %s""",
                    (new_type, Jsonb(merged_meta), importance, memory_id),
                )

        if content is not None:
            self.entangle_new(memory_id)
        return True

    def delete(self, memory_id: str) -> bool:
        """Delete a memory and its links, on explicit request only."""
        with self.pool.connection() as conn:
            row = conn.execute(
                "DELETE FROM memory_nodes WHERE memory_id = %s RETURNING memory_id",
                (memory_id,),
            ).fetchone()
        return row is not None

    # -- reading -----------------------------------------------------------

    def recall(self,
               query: str,
               limit: int = 5,
               memory_types: Optional[Sequence[str]] = None,
               project: Optional[str] = None,
               min_importance: float = 0.0,
               entanglement_boost: bool = True,
               candidate_factor: int = 4,
               touch: bool = True) -> List[Recall]:
        """
        Recall memories relevant to a query.

        relevance = fidelity * importance_weight. Entangled neighbours of a
        strong hit get a boost, so recalling one memory surfaces what it is
        connected to.
        """
        embedding = self.encoder.embed(query).tolist()
        query_state = self.encoder.quantum_state(query, "semantic")

        filters, params = ["embedding IS NOT NULL"], []
        if memory_types:
            filters.append("memory_type = ANY(%s)")
            params.append(list(memory_types))
        if project:
            filters.append("project = %s")
            params.append(project)
        if min_importance > 0:
            filters.append("importance >= %s")
            params.append(min_importance)
        where = " AND ".join(filters)

        with self.pool.connection() as conn:
            rows = self._nearest(
                conn,
                f"""SELECT n.memory_id, n.content, n.content_text, n.memory_type, n.quantum_state,
                           n.importance, n.access_count, n.last_accessed, n.creation_time,
                           n.metadata, 1 - (n.embedding <=> %s::vector) AS cosine
                      FROM (SELECT memory_id FROM memory_nodes
                             WHERE {where}
                          ORDER BY embedding <=> %s::vector
                             LIMIT %s) nearest
                      JOIN memory_nodes n USING (memory_id)
                  ORDER BY cosine DESC""",
                [embedding, *params, embedding, max(limit * candidate_factor, limit)],
            )

            now = datetime.now(timezone.utc)
            scored: Dict[str, Recall] = {}
            for r in rows:
                fidelity = self.encoder.fidelity(
                    query_state, self.encoder.unpack_state(r["quantum_state"])
                )
                relevance = fidelity * _importance_weight(r["importance"])

                scored[r["memory_id"]] = Recall(
                    memory_id=r["memory_id"],
                    content=(r["content"] or {}).get("value"),
                    memory_type=r["memory_type"],
                    similarity=float(r["cosine"]),
                    fidelity=fidelity,
                    relevance=relevance,
                    importance=float(r["importance"]),
                    access_count=r["access_count"],
                    last_accessed=r["last_accessed"],
                    created=r["creation_time"],
                    metadata=r["metadata"] or {},
                )

            if entanglement_boost and scored:
                self._apply_entanglement_boost(conn, scored, query_state, now)

            results = sorted(scored.values(), key=lambda x: x.relevance, reverse=True)[:limit]

            if touch and results:
                conn.execute(
                    """UPDATE memory_nodes
                          SET access_count = access_count + 1, last_accessed = now()
                        WHERE memory_id = ANY(%s)""",
                    ([r.memory_id for r in results],),
                )
        return results

    def _apply_entanglement_boost(self, conn, scored: Dict[str, Recall],
                                  query_state: np.ndarray, now: datetime) -> None:
        """Pull in neighbours of strong hits; a memory's links are part of it."""
        seeds = [m for m, r in scored.items() if r.relevance > 0.1]
        if not seeds:
            return

        neighbours = conn.execute(
            """SELECT e.from_memory, e.to_memory, e.strength,
                      n.content, n.memory_type, n.quantum_state, n.importance,
                      n.access_count, n.last_accessed, n.creation_time, n.metadata
                 FROM entanglement_edges e
                 JOIN memory_nodes n ON n.memory_id = e.to_memory
                WHERE e.from_memory = ANY(%s) AND e.strength >= %s""",
            (seeds, MODERATE),
        ).fetchall()

        for nb in neighbours:
            seed = scored.get(nb["from_memory"])
            if seed is None:
                continue
            boost = seed.relevance * float(nb["strength"]) * 0.5

            target = scored.get(nb["to_memory"])
            if target is not None:
                target.relevance += boost
                continue

            fidelity = self.encoder.fidelity(
                query_state, self.encoder.unpack_state(nb["quantum_state"])
            )
            scored[nb["to_memory"]] = Recall(
                memory_id=nb["to_memory"],
                content=(nb["content"] or {}).get("value"),
                memory_type=nb["memory_type"],
                similarity=0.0,
                fidelity=fidelity,
                relevance=fidelity * _importance_weight(nb["importance"]) + boost,
                importance=float(nb["importance"]),
                access_count=nb["access_count"],
                last_accessed=nb["last_accessed"],
                created=nb["creation_time"],
                metadata=nb["metadata"] or {},
            )

    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        with self.cursor() as cur:
            row = cur.execute(
                """SELECT memory_id, content, memory_type, importance, access_count,
                          last_accessed, creation_time, metadata, project, session_id
                     FROM memory_nodes WHERE memory_id = %s""",
                (memory_id,),
            ).fetchone()
        if row:
            row["content"] = (row["content"] or {}).get("value")
        return row

    def neighbours(self, memory_id: str, min_strength: float = WEAK) -> List[Dict[str, Any]]:
        with self.cursor() as cur:
            rows = cur.execute(
                """SELECT e.to_memory, e.strength, n.content, n.memory_type
                     FROM entanglement_edges e
                     JOIN memory_nodes n ON n.memory_id = e.to_memory
                    WHERE e.from_memory = %s AND e.strength >= %s
                 ORDER BY e.strength DESC""",
                (memory_id, min_strength),
            ).fetchall()
        for r in rows:
            r["content"] = (r["content"] or {}).get("value")
        return rows

    def stats(self) -> Dict[str, Any]:
        with self.cursor() as cur:
            # fetchone() is typed Row | None; COUNT/MIN always return a row,
            # but the stub does not, so default before subscripting.
            total = (cur.execute(
                "SELECT count(*) AS n FROM memory_nodes"
            ).fetchone() or {"n": 0})["n"]
            by_type = cur.execute(
                "SELECT memory_type, count(*) AS n FROM memory_nodes GROUP BY memory_type"
            ).fetchall()
            ent = (cur.execute(
                "SELECT count(*) AS n FROM entanglements"
            ).fetchone() or {"n": 0})["n"]
            clusters = (cur.execute(
                "SELECT count(*) AS n FROM entanglement_clusters"
            ).fetchone() or {"n": 0})["n"]
            oldest = (cur.execute(
                "SELECT min(creation_time) AS t FROM memory_nodes"
            ).fetchone() or {"t": None})["t"]
        return {
            "total_memories": total,
            "by_type": {r["memory_type"]: r["n"] for r in by_type},
            "entanglements": ent,
            "clusters": clusters,
            "oldest_memory": oldest.isoformat() if oldest else None,
            "embedding_model_active": self.encoder.using_model,
        }


def _importance_weight(importance: float) -> float:
    """
    Bounded importance multiplier in (0, 1].

    The original used (importance + 1) / 5 over a 0-4 enum, so importance could
    move relevance by at most 5x and never outrank a much better semantic match.
    Using raw importance here let a 3.0 beat a 2.0 on a worse match -- measured:
    "what does Brion do for work" returned a memory at fidelity 0.213 above one
    at 0.263 purely on importance. Saturating keeps importance a tiebreaker.
    """
    imp = max(float(importance), 0.0)
    return imp / (1.0 + imp)


def _jaccard(text_a: str, text_b: str) -> float:
    """Word-set overlap, as _calculate_semantic_similarity in the original."""
    a = set(text_a.lower().split())
    b = set(text_b.lower().split())
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
