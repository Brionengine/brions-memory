-- Brion's Memory — cloud schema
-- Ported from /mnt/c/quantum_brian/quantum_entanglement_memory.py
--
-- Target: DigitalOcean Managed Postgres. The 3 AMD droplets are stateless
-- compute running the encoder, entanglement and maintenance passes against
-- this one shared store, so a memory written on any node is visible to all.
--
-- Two representations are kept per memory, deliberately:
--
--   embedding      real-valued semantic vector -> pgvector HNSW, does ANN recall
--   quantum_state  the complex state, exact bytes -> fidelity |<s1|s2>|^2
--
-- pgvector is real-valued and cannot express a complex amplitude, and the
-- entanglement math needs the phase. So retrieval is two-stage: Postgres
-- narrows to candidates by embedding, the worker computes exact quantum
-- overlap on those. Neither representation is derivable from the other.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ---------------------------------------------------------------------------
-- Enums — values match MemoryType / EntanglementStrength in the source
-- ---------------------------------------------------------------------------

DO $$ BEGIN
    CREATE TYPE memory_type AS ENUM (
        'semantic',     -- conceptual knowledge
        'episodic',     -- experience-based
        'procedural',   -- skills and processes
        'associative',  -- connection-based
        'emotional',    -- emotion-linked
        'contextual',   -- context-dependent
        'meta',         -- meta-cognitive
        'quantum'       -- pure quantum state
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- ---------------------------------------------------------------------------
-- Memory nodes — one row per QuantumMemoryNode
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS memory_nodes (
    memory_id          text PRIMARY KEY,
    content            jsonb       NOT NULL,
    content_text       text        NOT NULL,
    memory_type        memory_type NOT NULL,

    -- Semantic embedding from the local model on the AMD workers.
    -- 384 = all-MiniLM-L6-v2. Change here and in encoder config together.
    embedding          vector(384),

    -- Exact complex state (numpy complex128 .tobytes()). Not indexable and not
    -- meant to be — it exists so fidelity stays exact after a round trip.
    quantum_state      bytea       NOT NULL,
    quantum_dimension  integer     NOT NULL,

    creation_time      timestamptz NOT NULL DEFAULT now(),
    last_accessed      timestamptz NOT NULL DEFAULT now(),
    access_count       integer     NOT NULL DEFAULT 0,
    importance         real        NOT NULL DEFAULT 1.0,
    quantum_signature  text        NOT NULL,
    metadata           jsonb       NOT NULL DEFAULT '{}'::jsonb,

    -- Which session wrote this, and which node. Needed once three servers
    -- write concurrently and a memory has to say where it came from.
    session_id         text,
    origin_node        text,
    project            text,

    CONSTRAINT memory_nodes_importance_nonneg CHECK (importance >= 0)
);

-- The source uses quantum_signature as a content-hash dedupe index.
CREATE UNIQUE INDEX IF NOT EXISTS memory_nodes_signature
    ON memory_nodes (quantum_signature);

-- ---------------------------------------------------------------------------
-- Entanglement graph — QuantumMemoryNode.entangled_memories, normalised
-- ---------------------------------------------------------------------------
--
-- In the source this is a dict on each node, written to both sides on every
-- link. Storing one row per unordered pair makes that symmetry a property of
-- the schema instead of something the code has to remember to maintain.

CREATE TABLE IF NOT EXISTS entanglements (
    memory_a    text NOT NULL REFERENCES memory_nodes(memory_id) ON DELETE CASCADE,
    memory_b    text NOT NULL REFERENCES memory_nodes(memory_id) ON DELETE CASCADE,
    strength    real NOT NULL CHECK (strength BETWEEN 0 AND 1),
    quantum_overlap     real,
    semantic_similarity real,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now(),

    -- Canonical ordering: one row per pair, no (a,b)/(b,a) duplicates.
    CONSTRAINT entanglements_ordered CHECK (memory_a < memory_b),
    PRIMARY KEY (memory_a, memory_b)
);

CREATE INDEX IF NOT EXISTS entanglements_a        ON entanglements (memory_a, strength DESC);
CREATE INDEX IF NOT EXISTS entanglements_b        ON entanglements (memory_b, strength DESC);
CREATE INDEX IF NOT EXISTS entanglements_strength ON entanglements (strength DESC);

-- Read either direction without the caller knowing the canonical order.
CREATE OR REPLACE VIEW entanglement_edges AS
    SELECT memory_a AS from_memory, memory_b AS to_memory, strength,
           quantum_overlap, semantic_similarity, created_at, updated_at
    FROM entanglements
    UNION ALL
    SELECT memory_b, memory_a, strength,
           quantum_overlap, semantic_similarity, created_at, updated_at
    FROM entanglements;

-- ---------------------------------------------------------------------------
-- Entanglement clusters
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS entanglement_clusters (
    cluster_id        text PRIMARY KEY,
    cluster_state     bytea       NOT NULL,
    cluster_strength  real        NOT NULL DEFAULT 0,
    cluster_type      text        NOT NULL DEFAULT 'mixed',
    creation_time     timestamptz NOT NULL DEFAULT now(),
    last_update       timestamptz NOT NULL DEFAULT now()
);

-- entanglement_matrix is NOT stored. It is derivable from the members' pairwise
-- strengths in `entanglements`, and a stored copy is one more thing that can
-- silently disagree with the graph it summarises.
CREATE TABLE IF NOT EXISTS cluster_members (
    cluster_id  text NOT NULL REFERENCES entanglement_clusters(cluster_id) ON DELETE CASCADE,
    memory_id   text NOT NULL REFERENCES memory_nodes(memory_id) ON DELETE CASCADE,
    added_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (cluster_id, memory_id)
);

CREATE INDEX IF NOT EXISTS cluster_members_memory ON cluster_members (memory_id);

-- ---------------------------------------------------------------------------
-- Sessions — from session_memory_manager.py
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS sessions (
    session_id    text PRIMARY KEY,
    user_id       text,
    project_path  text,
    session_type  text NOT NULL DEFAULT 'coding',
    privacy       text NOT NULL DEFAULT 'private'
                       CHECK (privacy IN ('private','shared','public')),
    started_at    timestamptz NOT NULL DEFAULT now(),
    last_active   timestamptz NOT NULL DEFAULT now(),
    ended_at      timestamptz,
    summary       text,
    metadata      jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS sessions_active ON sessions (last_active DESC);
CREATE INDEX IF NOT EXISTS memory_nodes_session ON memory_nodes (session_id);

-- ---------------------------------------------------------------------------
-- Indexes for recall
-- ---------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS memory_nodes_embedding_hnsw
    ON memory_nodes USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS memory_nodes_type       ON memory_nodes (memory_type);
CREATE INDEX IF NOT EXISTS memory_nodes_importance ON memory_nodes (importance DESC);
CREATE INDEX IF NOT EXISTS memory_nodes_created    ON memory_nodes (creation_time DESC);
CREATE INDEX IF NOT EXISTS memory_nodes_project    ON memory_nodes (project) WHERE project IS NOT NULL;
CREATE INDEX IF NOT EXISTS memory_nodes_fts        ON memory_nodes USING gin (to_tsvector('english', content_text));
CREATE INDEX IF NOT EXISTS memory_nodes_trgm       ON memory_nodes USING gin (content_text gin_trgm_ops);
