-- =====================================================================
-- MagmaAssistance -- PostgreSQL Schema
-- =====================================================================
-- Replaces the old single flat `audit_log` table with the normalized
-- design from the "Saving History in Postgres DataBase" area of the
-- architecture diagram (Untitled-2026-08-03-1515.excalidraw):
--
--   OVERVIEW      -- one row per turn (prompt/output), the unique key
--                    everything else hangs off of
--   DETAILS       -- depth info per OVERVIEW row: how many times the
--                    agent tried, tokens used
--   TOOLS         -- master registry of tools currently available
--   TOOLS_SEC     -- per-OVERVIEW-row record of which tool ran and
--                    whether it worked (aggregate success/fail counts
--                    with a GROUP BY over this table, rather than a
--                    running counter, so it can't drift out of sync)
--   TOOLS_DETAILS -- depth view per TOOLS_SEC row: input given / output
--                    received from that tool call
--   TOKEN_DETAILS -- per-department token usage & allotment, for both
--                    the Agent and TTS
--
-- Intentionally NOT included (marked red / "not developed" / "table
-- schema is pending" in the diagram -- add these once designed):
--   OCR_DETAILS
--
-- sessions and file_uploads are unchanged -- they're not part of the
-- diagram's DB Area, and file_uploads is still required by
-- storage/s3_storage.py and the /api/upload-document and /api/upload-document
-- endpoints in server.py.
--
-- Safe to run multiple times (idempotent: IF NOT EXISTS / DO blocks).
-- Run with: psql -h $PGHOST -p $PGPORT -U $PGUSER -d $PGDATABASE -f db/schema.sql
-- or via:   python db/init_db.py
-- =====================================================================

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;   -- gen_random_uuid()

-- ---------------------------------------------------------------------
-- Enum types (created idempotently -- CREATE TYPE has no IF NOT EXISTS)
-- ---------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'turn_role') THEN
        CREATE TYPE turn_role AS ENUM ('user', 'assistant', 'tool', 'system');
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'tool_status') THEN
        CREATE TYPE tool_status AS ENUM ('success', 'error', 'not_found');
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'upload_kind') THEN
        CREATE TYPE upload_kind AS ENUM ('purchase_order', 'general_document', 'audio');
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'upload_status') THEN
        CREATE TYPE upload_status AS ENUM ('pending', 'processing', 'processed', 'failed');
    END IF;
END $$;

-- ---------------------------------------------------------------------
-- sessions -- one row per conversation thread (unchanged)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    user_id         TEXT,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_active_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions (user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_last_active ON sessions (last_active_at DESC);

-- ---------------------------------------------------------------------
-- Drop the old flat audit table -- replaced by OVERVIEW / DETAILS /
-- TOOLS_SEC / TOOLS_DETAILS below. No migration of old rows: nothing in
-- the new design preserves audit_log's shape 1:1, and this project's
-- audit trail is disposable/regenerable, so this is a clean cut rather
-- than a column-by-column migration.
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS audit_log CASCADE;

-- ---------------------------------------------------------------------
-- OVERVIEW -- one row per turn: prompt + output, with a unique key
-- ("Basic table of Prompt and output of model with date and time")
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS overview (
    id                BIGSERIAL PRIMARY KEY,
    session_id        TEXT NOT NULL REFERENCES sessions (session_id) ON DELETE CASCADE,

    role              turn_role NOT NULL,           -- 'user' | 'assistant' | 'tool' | 'system'
    user_id           TEXT,                         -- who prompted the underlying request
    prompt_text       TEXT,                         -- the user question this row answers
    output_text       TEXT NOT NULL,                -- message text / tool result (stringified)

    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_overview_session      ON overview (session_id, id);
CREATE INDEX IF NOT EXISTS idx_overview_created_at   ON overview (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_overview_user_id      ON overview (user_id) WHERE user_id IS NOT NULL;

-- ---------------------------------------------------------------------
-- DETAILS -- depth details of the prompt: how many times the agent
-- tried, and tokens used by the user
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS details (
    id                BIGSERIAL PRIMARY KEY,
    overview_id       BIGINT NOT NULL REFERENCES overview (id) ON DELETE CASCADE,

    tries             INTEGER,
    tokens_used       INTEGER,

    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (overview_id)
);

-- ---------------------------------------------------------------------
-- TOOLS -- master registry of tools currently present with us
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tools (
    tool_id           SERIAL PRIMARY KEY,
    tool_name         TEXT NOT NULL UNIQUE,
    description       TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------
-- TOOLS_SEC -- with the OVERVIEW unique key, which tool(s) were used
-- by the AI and whether each call worked or failed
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tools_sec (
    id                BIGSERIAL PRIMARY KEY,
    overview_id       BIGINT NOT NULL REFERENCES overview (id) ON DELETE CASCADE,
    tool_id           INTEGER NOT NULL REFERENCES tools (tool_id),

    status            tool_status NOT NULL,         -- success | error | not_found
    duration_ms       INTEGER,                      -- tool execution time

    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_tools_sec_overview     ON tools_sec (overview_id);
CREATE INDEX IF NOT EXISTS idx_tools_sec_tool_status  ON tools_sec (tool_id, status, created_at DESC);

-- ---------------------------------------------------------------------
-- TOOLS_DETAILS -- depth view per tool call: what input was given and
-- what output was received from the tool
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tools_details (
    id                BIGSERIAL PRIMARY KEY,
    tools_sec_id      BIGINT NOT NULL REFERENCES tools_sec (id) ON DELETE CASCADE,

    tool_args         JSONB,                        -- input given to the tool
    tool_output       TEXT,                          -- output received from the tool
    error_message     TEXT,                          -- populated when tools_sec.status = 'error'

    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (tools_sec_id)
);

-- ---------------------------------------------------------------------
-- TOKEN_DETAILS -- tokens used by each department, and total tokens
-- left for both the Agent and TTS
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS token_details (
    id                     BIGSERIAL PRIMARY KEY,
    department             TEXT NOT NULL UNIQUE,

    agent_tokens_allotted  BIGINT,
    agent_tokens_used      BIGINT NOT NULL DEFAULT 0,
    tts_tokens_allotted    BIGINT,
    tts_tokens_used        BIGINT NOT NULL DEFAULT 0,

    updated_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE OR REPLACE FUNCTION touch_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_token_details_touch ON token_details;
CREATE TRIGGER trg_token_details_touch
    BEFORE UPDATE ON token_details
    FOR EACH ROW
    EXECUTE FUNCTION touch_updated_at();

-- ---------------------------------------------------------------------
-- file_uploads -- metadata for every file uploaded, content lives in S3
-- (unchanged -- not part of the diagram's DB Area, still required by
-- storage/s3_storage.py and the upload endpoints in server.py)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS file_uploads (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id          TEXT REFERENCES sessions (session_id) ON DELETE SET NULL,
    user_id             TEXT,
    uploaded_at         TIMESTAMPTZ NOT NULL DEFAULT now(),

    original_filename   TEXT NOT NULL,
    content_type        TEXT NOT NULL,
    file_size_bytes     BIGINT NOT NULL CHECK (file_size_bytes >= 0),
    checksum_sha256     TEXT NOT NULL,

    upload_kind         upload_kind NOT NULL,
    status              upload_status NOT NULL DEFAULT 'pending',

    s3_bucket           TEXT NOT NULL,
    s3_key              TEXT NOT NULL,
    s3_region           TEXT NOT NULL,
    s3_version_id       TEXT,

    extracted_metadata  JSONB NOT NULL DEFAULT '{}'::jsonb,
    processing_error    TEXT,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (s3_bucket, s3_key)
);

CREATE INDEX IF NOT EXISTS idx_file_uploads_session   ON file_uploads (session_id);
CREATE INDEX IF NOT EXISTS idx_file_uploads_user      ON file_uploads (user_id);
CREATE INDEX IF NOT EXISTS idx_file_uploads_uploaded  ON file_uploads (uploaded_at DESC);
CREATE INDEX IF NOT EXISTS idx_file_uploads_checksum  ON file_uploads (checksum_sha256);

DROP TRIGGER IF EXISTS trg_file_uploads_touch ON file_uploads;
CREATE TRIGGER trg_file_uploads_touch
    BEFORE UPDATE ON file_uploads
    FOR EACH ROW
    EXECUTE FUNCTION touch_updated_at();

COMMIT;
-- ---------------------------------------------------------------------
-- LONG_TERM_MEMORY -- LangGraph Memory Store backend
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS long_term_memory (
    id                BIGSERIAL PRIMARY KEY,
    user_id           TEXT NOT NULL,
    memory_type       TEXT NOT NULL, -- 'semantic', 'episodic', 'procedural'
    memory_key        TEXT NOT NULL,
    memory_value      TEXT NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, memory_type, memory_key)
);

CREATE INDEX IF NOT EXISTS idx_ltm_user ON long_term_memory (user_id, memory_type);

DROP TRIGGER IF EXISTS trg_ltm_touch ON long_term_memory;
CREATE TRIGGER trg_ltm_touch
    BEFORE UPDATE ON long_term_memory
    FOR EACH ROW
    EXECUTE FUNCTION touch_updated_at();
