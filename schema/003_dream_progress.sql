-- Schema migration 003: dream-heavy progress tracking
-- Adds nullable columns to dream_runs that heavy.py updates as each phase progresses.
-- Idempotency is provided by the schema_versions tracker (see src/hippo/storage/migrations.py).

ALTER TABLE dream_runs ADD COLUMN current_phase TEXT;
ALTER TABLE dream_runs ADD COLUMN phase_done INTEGER;
ALTER TABLE dream_runs ADD COLUMN phase_total INTEGER;
ALTER TABLE dream_runs ADD COLUMN phase_started_at INTEGER;
ALTER TABLE dream_runs ADD COLUMN last_progress_at INTEGER;
