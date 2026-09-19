# Brion's Memory

Long-term memory for every AI Brion runs, stored in the cloud and reachable three ways
(hooks, MCP, a Python client — see *How an AI reaches memory*).

Built on the quantum entanglement memory architecture from `/mnt/c/quantum_brian`
(July 2025): memories are nodes with a quantum state, related memories entangle,
recall pulls in what a strong match is entangled with. Memories are permanent.

## Layout

| path | what |
|---|---|
| `brions_memory/encoder.py` | 384-D semantic embedding + complex quantum state. Fidelity = cos² exactly. |
| `brions_memory/store.py` | Postgres store: recall, entanglement |
| `brions_memory/mcp_server.py` | MCP over stdio. Tools: `remember`, `recall`, `related`, `memory_stats` |
| `brions_memory/memory_client.py` | Stdlib-only recall client for AIs that are not MCP clients |
| `sql/001_schema.sql` | Cloud schema |
| `migrations/import_history.py` | Imports claude-mem + the Dec 2025 system; redacts credentials |

## Where things run

- **Database**: DO Managed Postgres 17, `brions-memory`, nyc3, pgvector 0.8.6
- **MCP server**: launched by Claude Code on demand, user scope

## How an AI reaches memory

Three routes, so every AI Brion runs can remember, not only the one in front of him.

| Route | For | Reaches memory |
|---|---|---|
| **Hooks** (`hooks/memory_hook.py`) | Claude Code, every project | Automatically, before the agent reads the request — `SessionStart`, `UserPromptSubmit`, and `SubagentStart`, so subagents start informed too |
| **MCP server** (`brions_memory/mcp_server.py`) | Any MCP client (Claude Code, Claude Desktop, other agents) | When the AI chooses to call `recall` / `remember` |
| **Python client** (`brions_memory/memory_client.py`) | Everything else — local quantum agents, Brian, scheduled jobs | `from brions_memory.memory_client import context_block` |

```python
sys.path.insert(0, "/mnt/c/Brion's Memory")
from brions_memory.memory_client import context_block
block = context_block("what accuracy did the 20-qubit WDBC model reach?")
if block:                       # None when memory is unreachable; never raises
    system_prompt = block + "\n\n" + system_prompt
```

**Audience is the constraint, not capability.** The store holds personal and
infrastructure memories and has no per-visitor scoping, so anything answering people
other than Brion (a public endpoint, a shared agent) must gate recall behind its own
owner check — `memory_client.audience_is_owner()` is the explicit form. Brian's public
chat at brionquantum.com is the live example: he must not inherit this by default.

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
