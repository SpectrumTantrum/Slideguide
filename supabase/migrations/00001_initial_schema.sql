-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Enums (matching Prisma schema exactly)
CREATE TYPE upload_status AS ENUM ('PROCESSING', 'READY', 'ERROR');
CREATE TYPE session_phase AS ENUM ('GREETING', 'TOPIC_SELECTION', 'TEACHING', 'QUIZ', 'REVIEW', 'WRAP_UP');
CREATE TYPE message_role AS ENUM ('USER', 'ASSISTANT', 'SYSTEM', 'TOOL');

-- Uploads table (was Prisma Upload model)
CREATE TABLE uploads (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    status upload_status NOT NULL DEFAULT 'PROCESSING',
    total_slides INTEGER NOT NULL DEFAULT 0,
    storage_path TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Slides table (was Prisma Slide model)
CREATE TABLE slides (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    upload_id TEXT NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
    slide_number INTEGER NOT NULL,
    title TEXT,
    text_content TEXT NOT NULL,
    has_images BOOLEAN NOT NULL DEFAULT false,
    image_paths JSONB,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(upload_id, slide_number)
);

-- Sessions table (was Prisma Session model)
CREATE TABLE sessions (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    upload_id TEXT NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
    student_id TEXT,
    phase session_phase NOT NULL DEFAULT 'GREETING',
    metadata JSONB,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at TIMESTAMPTZ
);

-- Messages table (was Prisma Message model)
CREATE TABLE messages (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role message_role NOT NULL,
    content TEXT NOT NULL,
    tool_calls JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Student progress table (was Prisma StudentProgress model)
CREATE TABLE student_progress (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    session_id TEXT NOT NULL UNIQUE REFERENCES sessions(id) ON DELETE CASCADE,
    upload_id TEXT NOT NULL,
    topics_covered JSONB NOT NULL DEFAULT '[]'::jsonb,
    quiz_scores JSONB NOT NULL DEFAULT '{}'::jsonb,
    total_questions INTEGER NOT NULL DEFAULT 0,
    correct_answers INTEGER NOT NULL DEFAULT 0,
    confidence_level DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    last_active TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Cost logs table (was Prisma CostLog model)
CREATE TABLE cost_logs (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd DOUBLE PRECISION NOT NULL,
    operation TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Auto-update triggers
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS trigger AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER uploads_updated_at BEFORE UPDATE ON uploads
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER student_progress_updated_at BEFORE UPDATE ON student_progress
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- Indexes
CREATE INDEX idx_slides_upload_id ON slides(upload_id);
CREATE INDEX idx_sessions_upload_id ON sessions(upload_id);
CREATE INDEX idx_messages_session_id ON messages(session_id);
CREATE INDEX idx_messages_created_at ON messages(session_id, created_at);
CREATE INDEX idx_student_progress_session_id ON student_progress(session_id);
CREATE INDEX idx_cost_logs_session_id ON cost_logs(session_id);
