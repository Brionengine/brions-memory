#!/usr/bin/env python3
"""
Brion's Memory — MCP server.

Speaks MCP over stdio as raw JSON-RPC 2.0 with no SDK dependency, because this
has to start reliably on three droplets and a laptop, and every dependency is
one more thing that can be missing at 3am.

The previous mcp_claude_memory_server.py was named for MCP but never spoke it:
no stdio loop, no initialize / tools/list / tools/call. Claude Code could not
have loaded it. This one implements the actual protocol.

Register with:
    claude mcp add brions-memory -- /path/to/.venv/bin/python -m brions_memory.mcp_server

Protocol notes:
  - stdout carries JSON-RPC and nothing else. Every log line goes to stderr;
    one stray print() to stdout corrupts the stream and the server looks dead.
  - notifications (no "id") get no response, ever.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional

logging.basicConfig(
    level=os.environ.get("BRIONS_MEMORY_LOG", "INFO"),
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("brions-memory")

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "brions-memory", "version": "1.0.0"}

_store = None


def get_store():
    """Open the store on first use so a missing DB fails a call, not startup."""
    global _store
    if _store is None:
        from .encoder import Encoder
        from .store import MemoryStore
        model = os.environ.get("BRIONS_MEMORY_EMBED_MODEL")
        encoder = Encoder(model_name=model) if model else Encoder()
        _store = MemoryStore(encoder=encoder)
        logger.info("Memory store connected (embeddings=%s)", encoder.using_model)
    return _store


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "remember",
        "description": (
            "Store something in Brion's long-term memory. Use for facts about Brion, "
            "decisions and why they were made, measured results, and anything that "
            "should survive this conversation. Storing the same thing twice "
            "strengthens the existing memory rather than duplicating it."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "What to remember, in full sentences."},
                "memory_type": {
                    "type": "string",
                    "enum": ["semantic", "episodic", "procedural", "associative",
                             "emotional", "contextual", "meta", "quantum"],
                    "default": "semantic",
                    "description": (
                        "semantic=facts and knowledge, episodic=things that happened, "
                        "procedural=how to do something, emotional=how Brion felt about "
                        "something, contextual=true only in a particular context, "
                        "meta=about the memory system itself."
                    ),
                },
                "importance": {"type": "number", "default": 1.0,
                               "description": "Higher ranks above equally good matches. 1.0 is normal."},
                "project": {"type": "string", "description": "Project this belongs to, if any."},
                "metadata": {"type": "object", "description": "Any structured extras."},
            },
            "required": ["content"],
        },
    },
    {
        "name": "recall",
        "description": (
            "Search Brion's long-term memory. Returns memories ranked by semantic "
            "fidelity and importance, and pulls in memories entangled with "
            "strong matches. Use before asking Brion something he may have already told you."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What you are trying to remember."},
                "limit": {"type": "integer", "default": 5},
                "memory_types": {"type": "array", "items": {"type": "string"},
                                 "description": "Restrict to these types."},
                "project": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "related",
        "description": (
            "Given a memory id, return the memories entangled with it, strongest first. "
            "Use to follow a thread outward from something already recalled."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
                "min_strength": {"type": "number", "default": 0.3},
            },
            "required": ["memory_id"],
        },
    },
    {
        "name": "memory_stats",
        "description": (
            "Health and size of the memory store: how many memories by type, how many "
            "entanglements and clusters, oldest memory, and whether real embeddings are "
            "active or it has fallen back to token hashing."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def tool_remember(args: Dict[str, Any]) -> str:
    store = get_store()
    memory_id = store.store(
        content=args["content"],
        memory_type=args.get("memory_type", "semantic"),
        importance=float(args.get("importance", 1.0)),
        metadata=args.get("metadata"),
        project=args.get("project"),
        session_id=os.environ.get("BRIONS_MEMORY_SESSION"),
    )
    return f"Remembered as {memory_id}."


def tool_recall(args: Dict[str, Any]) -> str:
    store = get_store()
    results = store.recall(
        query=args["query"],
        limit=int(args.get("limit", 5)),
        memory_types=args.get("memory_types"),
        project=args.get("project"),
    )
    if not results:
        return "No memories found for that query."

    lines = []
    for r in results:
        when = r.created.strftime("%Y-%m-%d") if r.created else "unknown"
        lines.append(
            f"[{r.memory_id}] ({r.memory_type}, {when}, "
            f"relevance {r.relevance:.3f}, fidelity {r.fidelity:.3f})\n{r.content}"
        )
    return "\n\n".join(lines)


def tool_related(args: Dict[str, Any]) -> str:
    store = get_store()
    rows = store.neighbours(args["memory_id"], float(args.get("min_strength", 0.3)))
    if not rows:
        return "No entangled memories above that strength."
    return "\n\n".join(
        f"[{r['to_memory']}] ({r['memory_type']}, strength {r['strength']:.3f})\n{r['content']}"
        for r in rows
    )


def tool_memory_stats(args: Dict[str, Any]) -> str:
    return json.dumps(get_store().stats(), indent=2, default=str)


HANDLERS: Dict[str, Callable[[Dict[str, Any]], str]] = {
    "remember": tool_remember,
    "recall": tool_recall,
    "related": tool_related,
    "memory_stats": tool_memory_stats,
}


# ---------------------------------------------------------------------------
# JSON-RPC plumbing
# ---------------------------------------------------------------------------

def _result(request_id: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": payload}


def _error(request_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def handle(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params") or {}

    # A notification has no id and must never be answered.
    if request_id is None:
        logger.debug("notification: %s", method)
        return None

    if method == "initialize":
        return _result(request_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
        })

    if method == "tools/list":
        return _result(request_id, {"tools": TOOLS})

    if method == "tools/call":
        name = params.get("name")
        handler = HANDLERS.get(name) if isinstance(name, str) else None
        if handler is None:
            return _error(request_id, -32602, f"Unknown tool: {name}")
        try:
            raw_args = params.get("arguments") or {}
            arguments: Dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
            text = handler(arguments)
            return _result(request_id, {"content": [{"type": "text", "text": text}]})
        except Exception as exc:
            logger.error("tool %s failed: %s", name, traceback.format_exc())
            # Report the failure through the result channel so the model can
            # see it and adapt, rather than as a protocol error it cannot read.
            return _result(request_id, {
                "content": [{"type": "text", "text": f"Tool {name} failed: {exc}"}],
                "isError": True,
            })

    if method in ("ping",):
        return _result(request_id, {})

    return _error(request_id, -32601, f"Method not found: {method}")


def main() -> None:
    logger.info("Brion's Memory MCP server starting")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("could not parse line: %.120s", line)
            continue

        response = handle(message)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

    logger.info("stdin closed; exiting")


if __name__ == "__main__":
    main()
