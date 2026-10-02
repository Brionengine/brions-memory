-- Brion's Memory — archive of forgotten and overwritten memories.
--
-- forget and update_memory are MCP tools an LLM can call, and the model reads
-- untrusted text (web pages, recalled memories) that could ask it to. A model
-- supplied "confirm" flag authorises nothing, so instead nothing is destroyed:
-- forget moves the row here before deleting it, update_memory copies the old
-- version here before overwriting it, and restore_memory brings either back.
--
-- Same columns as memory_nodes (LIKE), but memory_id is not unique: one memory
-- can have many archived versions. Entanglements are not archived; they are
-- derived, and restore rebuilds them with entangle_new.

CREATE TABLE IF NOT EXISTS memory_archive (
    LIKE memory_nodes INCLUDING DEFAULTS,
    archive_id      bigserial   PRIMARY KEY,
    archived_at     timestamptz NOT NULL DEFAULT now(),
    archive_reason  text        NOT NULL CHECK (archive_reason IN ('forget', 'update', 'restore'))
);

CREATE INDEX IF NOT EXISTS memory_archive_memory ON memory_archive (memory_id, archived_at DESC);
