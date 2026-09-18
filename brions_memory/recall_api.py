#!/usr/bin/env python3
"""
Always-warm recall service. Runs on the AMD droplet.

The embedding model takes ~38s to load cold (measured locally 2026-09-13), so
it cannot be loaded per prompt. This process loads it once and answers recall
requests in milliseconds, which is what lets memory be injected automatically
before every prompt instead of being a tool the agent has to decide to call.

HTTPS with a self-signed certificate that clients pin, plus a bearer token.
Standard library server on purpose: no web framework to keep patched on a box
that also serves a public site.

    POST /recall   {"query": str, "limit": int, "min_fidelity": float}
    POST /profile  {"limit": int}      who Brion is + what happened recently
    GET  /health
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import ssl
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .encoder import Encoder
from .store import MemoryStore

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("brions-memory-recall")

TOKEN = os.environ["BRIONS_MEMORY_API_TOKEN"]
MAX_BODY = 64 * 1024

PROFILE_QUERIES = (
    "who Brion is, his work, background and what he cares about",
    "how Brion prefers to work and communicate, his rules and preferences",
)

store: MemoryStore
_ready = threading.Event()


def _serialise(results):
    return [{
        "id": r.memory_id,
        "type": r.memory_type,
        "date": r.created.date().isoformat() if r.created else None,
        "fidelity": round(r.fidelity, 3),
        "relevance": round(r.relevance, 3),
        "content": r.content if isinstance(r.content, str) else str(r.content),
    } for r in results]


def do_recall(body: dict[str, Any]) -> dict[str, Any]:
    query = str(body.get("query", ""))[:4000]
    if not query.strip():
        return {"memories": []}
    limit = max(1, min(int(body.get("limit", 6)), 20))
    min_fid = float(body.get("min_fidelity", 0.0))
    results = store.recall(query, limit=limit, touch=False)
    return {"memories": _serialise([r for r in results if r.fidelity >= min_fid])}


def do_profile(body: dict[str, Any]) -> dict[str, Any]:
    limit = max(1, min(int(body.get("limit", 6)), 20))
    seen, about = set(), []
    for q in PROFILE_QUERIES:
        for r in store.recall(q, limit=limit, touch=False):
            if r.memory_id not in seen:
                seen.add(r.memory_id)
                about.append(r)
    about.sort(key=lambda r: r.relevance, reverse=True)

    with store.pool.connection() as conn:
        rows = conn.execute(
            """SELECT memory_id, memory_type, creation_time, content_text
                 FROM memory_nodes
                WHERE memory_type = 'episodic'
                  AND metadata->>'source' IN ('claude-mem/session_summaries', 'session')
             ORDER BY creation_time DESC
                LIMIT %s""",
            (max(1, min(int(body.get("recent", 4)), 10)),),
        ).fetchall()
    recent = [{
        "id": r["memory_id"], "type": r["memory_type"],
        "date": r["creation_time"].date().isoformat(),
        "content": r["content_text"],
    } for r in rows]
    return {"about": _serialise(about[:limit]), "recent": recent}


class Handler(BaseHTTPRequestHandler):
    server_version = "brions-memory"

    def log_message(self, format, *args):
        logger.info("%s %s", self.address_string(), format % args)

    def _send(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authorised(self):
        header = self.headers.get("Authorization", "")
        return hmac.compare_digest(header, f"Bearer {TOKEN}")

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": _ready.is_set()})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._authorised():
            return self._send(401, {"error": "unauthorised"})
        if not _ready.is_set():
            return self._send(503, {"error": "model loading"})
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_BODY:
            return self._send(413, {"error": "too large"})
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"error": "bad json"})

        started = time.time()
        try:
            if self.path == "/recall":
                result = do_recall(body)
            elif self.path == "/profile":
                result = do_profile(body)
            else:
                return self._send(404, {"error": "not found"})
        except Exception:
            logger.exception("request failed")
            return self._send(500, {"error": "internal"})
        result["ms"] = round((time.time() - started) * 1000, 1)
        self._send(200, result)


def main():
    global store
    host = os.environ.get("BRIONS_MEMORY_API_HOST", "0.0.0.0")
    port = int(os.environ.get("BRIONS_MEMORY_API_PORT", "8443"))

    store = MemoryStore(encoder=Encoder(allow_fallback=False), max_size=4)
    started = time.time()
    store.encoder.embed("warmup")
    _ready.set()
    logger.info("model warm in %.1fs", time.time() - started)

    httpd = ThreadingHTTPServer((host, port), Handler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(os.environ["BRIONS_MEMORY_TLS_CERT"], os.environ["BRIONS_MEMORY_TLS_KEY"])
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    logger.info("recall API listening on https://%s:%d", host, port)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
