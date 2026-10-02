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
| `brions_memory/mcp_server.py` | MCP over stdio. Memory: `remember`, `recall`, `related`, `get_memory`, `update_memory`, `forget`, `archived_memories`, `recent_memories`, `list_projects`, `memory_stats`. Quantum: `superposition_recall`, `quantum_fidelity`, `entanglement_path`, `build_clusters`, `clusters`, `cluster_members` |
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
| **Remote MCP** (`brions_memory/remote_mcp.py`) | ChatGPT and any remote MCP client | Over HTTPS with OAuth: ChatGPT's connector login asks for Brion's passphrase. Read tools + `remember` only |
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

Since 2026-10-02 Claude Code runs as `brion`, not root: the venv is `~/.local/share/brions-memory/venv` (Python 3.14), the MCP launcher is `~/.local/share/brions-memory/run-mcp.sh` (sources `.env`), and the hooks read `~/.config/brions-memory/client.json`. Paths in hooks and sync scripts resolve from `~`.

Re-registering the MCP server requires `PYTHONPATH="/mnt/c/Brion's Memory"`, or it only
works when Claude Code is started from this directory.

## Remote MCP for ChatGPT

ChatGPT cannot launch a local server; its connectors take a public HTTPS URL with OAuth or nothing. `remote_mcp.py` is both the OAuth 2.1 server (discovery, dynamic registration restricted to ChatGPT/OpenAI redirect hosts, PKCE S256, rotating refresh tokens, hashed token storage, serialised 5-try lockout, owner-opened registration window) and the MCP endpoint (`POST /mcp`). Deployed as `brions-memory-remote.service` behind Caddy (Let's Encrypt) on an OVH AMD server; config in `/etc/brions-memory/remote.env` (mode 640), tokens in `/var/lib/brions-memory`. Caddy's access log drops query codes, `Authorization` and `Location`. Passphrase hash: `python -m brions_memory.remote_mcp --hash-passphrase '<passphrase>'`.

Registration is closed by default — otherwise anyone could register their own connector and phish Brion with a link to the real login page. Open it right before connecting (it closes on the first successful login): on the server, `brions-memory-registration open` (15 min) / `close`.

In ChatGPT: Settings → Apps & Connectors → Advanced → Developer mode on → Create → URL `https://<host>/mcp`, Authentication **OAuth** → sign in with the passphrase.

Codex (CLI and the Windows desktop app) uses the local stdio server instead, plus the same recall hooks: `[mcp_servers.brions-memory]` in `config.toml` and `SessionStart`/`UserPromptSubmit` in `hooks.json`.

## Decisions worth not relitigating

- Memories are permanent (2026-09-13). Relevance is fidelity × importance weight and nothing
  time-based; importance never changes on its own and nothing deletes a memory except an explicit request.
- No tool call destroys a memory (2026-10-02). `forget` and `update_memory` are reachable by an LLM that reads untrusted text, and a model-supplied confirm flag authorises nothing, so both copy the old row to `memory_archive` (`sql/002_memory_archive.sql`) nothing is lost. Getting anything back out — reading archived text, restoring, or erasing for good (e.g. a credential that ended up in a memory) — is `python -m brions_memory.archive list|show|restore|search|purge`, human-only and deliberately not MCP: a model-reachable restore or listing would resurface or expose exactly what was forgotten.
- Memory type is a query filter, not a phase in the quantum state (both phase variants measured badly).
- Entanglement strength blends |cos|, not cos²; relevance uses importance/(1+importance).
