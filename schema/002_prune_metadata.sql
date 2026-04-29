-- Schema migration 002: prune-phase metadata
-- Idempotent — applied at most once per DB via the schema_versions tracker (see src/hippo/storage/migrations.py); IF NOT EXISTS for indexes.

-- Add last_reviewed_at to bodies (nullable; existing rows stay NULL → sort first in rolling slice)
-- sqlite has no "ADD COLUMN IF NOT EXISTS"; use a guarded approach via PRAGMA.
-- We rely on the migration runner's schema_versions check to prevent re-application,
-- so a plain ALTER is safe here.
ALTER TABLE bodies ADD COLUMN last_reviewed_at INTEGER;

-- Add bodies_archived_review counter to dream_runs
ALTER TABLE dream_runs ADD COLUMN bodies_archived_review INTEGER DEFAULT 0;

-- Index for the rolling slice query (active bodies ordered by review recency)
CREATE INDEX IF NOT EXISTS idx_bodies_review_queue
    ON bodies(last_reviewed_at) WHERE archived = 0;
