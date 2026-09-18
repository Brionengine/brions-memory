# Brion's Memory

Long-term memory for Claude Code, stored in the cloud and exposed as an MCP tool.

Built on the quantum entanglement memory architecture from `/mnt/c/quantum_brian`
(July 2025): memories are nodes with a quantum state, related memories entangle,
recall pulls in what a strong match is entangled with. Memories are permanent.

## Layout

| path | what |
|---|---|
| `brions_memory/encoder.py` | 384-D semantic embedding + complex quantum state. Fidelity = cos² exactly. |
| `brions_memory/store.py` | Postgres store: recall, entanglement |
| `brions_memory/mcp_server.py` | MCP over stdio. Tools: `remember`, `recall`, `related`, `memory_stats` |
| `sql/001_schema.sql` | Cloud schema |
| `migrations/import_history.py` | Imports claude-mem + the Dec 2025 system; redacts credentials |

## Where things run

- **Database**: DO Managed Postgres 17, `brions-memory`, nyc3, pgvector 0.8.6
- **MCP server**: launched by Claude Code on demand, user scope

## Operating

```bash
claude mcp list                                    # should show brions-memory ✔ Connected
set -a; . ./.env; set +a; psql "$BRIONS_MEMORY_DB_URL"
```

Re-registering the MCP server requires `PYTHONPATH="/mnt/c/Brion's Memory"`, or it only
works when Claude Code is started from this directory.

## Decisions worth not relitigating

- Memories are permanent (2026-09-13). Relevance is fidelity × importance weight and nothing
  time-based; importance never changes on its own and nothing deletes a memory except an explicit request.
- Memory type is a query filter, not a phase in the quantum state (both phase variants measured badly).
- Entanglement strength blends |cos|, not cos²; relevance uses importance/(1+importance).
