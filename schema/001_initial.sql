-- Schema migration 001: initial tables
-- Idempotent — every CREATE uses IF NOT EXISTS

CREATE TABLE IF NOT EXISTS bodies (
    body_id              TEXT PRIMARY KEY,
    file_path            TEXT NOT NULL,
    title                TEXT NOT NULL,
    scope                TEXT NOT NULL,
    archived             INTEGER NOT NULL DEFAULT 0,
    archive_reason       TEXT,
    archived_in_favor_of TEXT,
    source               TEXT NOT NULL,
    created_at           INTEGER NOT NULL,
    updated_at           INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bodies_scope_active ON bodies(scope) WHERE archived = 0;

CREATE TABLE IF NOT EXISTS heads (
    head_id           TEXT PRIMARY KEY,
    body_id           TEXT NOT NULL REFERENCES bodies(body_id) ON DELETE CASCADE,
    summary           TEXT NOT NULL,
    archived          INTEGER NOT NULL DEFAULT 0,
    archive_reason    TEXT,
    last_retrieved_at INTEGER,
    retrieval_count   INTEGER NOT NULL DEFAULT 0,
    created_at        INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_heads_body_active ON heads(body_id) WHERE archived = 0;

-- Vector store via sqlite-vec (1024 dims for mxbai-embed-large)
CREATE VIRTUAL TABLE IF NOT EXISTS head_embeddings USING vec0(
    head_id   TEXT PRIMARY KEY,
    embedding FLOAT[1024]
);

CREATE TABLE IF NOT EXISTS edges (
    edge_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    from_head  TEXT NOT NULL REFERENCES heads(head_id) ON DELETE CASCADE,
    to_head    TEXT NOT NULL REFERENCES heads(head_id) ON DELETE CASCADE,
    relation   TEXT NOT NULL,
    weight     REAL NOT NULL DEFAULT 1.0,
    created_at INTEGER NOT NULL,
    UNIQUE(from_head, to_head, relation)
);
CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_head);
CREATE INDEX IF NOT EXISTS idx_edges_to   ON edges(to_head);

CREATE TABLE IF NOT EXISTS capture_queue (
    queue_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT NOT NULL,
    project           TEXT,
    user_message      TEXT,
    assistant_message TEXT,
    transcript_path   TEXT,
    created_at        INTEGER NOT NULL,
    processed_at      INTEGER,
    processed_by_run  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_capture_unprocessed ON capture_queue(created_at) WHERE processed_at IS NULL;

CREATE TABLE IF NOT EXISTS turn_embeddings (
    turn_id    INTEGER PRIMARY KEY,
    capture_id INTEGER REFERENCES capture_queue(queue_id) ON DELETE CASCADE,
    summary    TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS turn_embeddings_vec USING vec0(
    turn_id   INTEGER PRIMARY KEY,
    embedding FLOAT[1024]
);

CREATE TABLE IF NOT EXISTS dream_runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    type          TEXT NOT NULL,
    started_at    INTEGER NOT NULL,
    completed_at  INTEGER,
    status        TEXT NOT NULL DEFAULT 'running',
    atoms_created INTEGER DEFAULT 0,
    heads_created INTEGER DEFAULT 0,
    edges_created INTEGER DEFAULT 0,
    contradictions_resolved INTEGER DEFAULT 0,
    error_message TEXT
);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_versions (
    version    INTEGER PRIMARY KEY,
    applied_at INTEGER NOT NULL
);
