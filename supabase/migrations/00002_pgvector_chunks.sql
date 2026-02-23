-- Slide chunks table for RAG (replaces ChromaDB)
CREATE TABLE slide_chunks (
    id TEXT PRIMARY KEY,
    upload_id TEXT NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
    slide_number INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    content_type TEXT NOT NULL DEFAULT 'text',
    content TEXT NOT NULL,
    embedding vector(1536),
    search_vector tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(content, '')), 'B')
    ) STORED,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(upload_id, slide_number, chunk_index)
);

-- HNSW index for cosine similarity
CREATE INDEX idx_chunks_embedding ON slide_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- GIN index for full-text search
CREATE INDEX idx_chunks_search_vector ON slide_chunks USING gin(search_vector);

CREATE INDEX idx_chunks_upload_id ON slide_chunks(upload_id);

-- Semantic search RPC function (called via supabase.rpc())
CREATE OR REPLACE FUNCTION match_slide_chunks(
    query_embedding vector(1536),
    filter_upload_id TEXT,
    match_count INTEGER DEFAULT 10,
    filter_slide_number INTEGER DEFAULT NULL
)
RETURNS TABLE (
    id TEXT,
    upload_id TEXT,
    slide_number INTEGER,
    chunk_index INTEGER,
    title TEXT,
    content_type TEXT,
    content TEXT,
    similarity FLOAT
)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        sc.id, sc.upload_id, sc.slide_number, sc.chunk_index,
        sc.title, sc.content_type, sc.content,
        1 - (sc.embedding <=> query_embedding) AS similarity
    FROM slide_chunks sc
    WHERE sc.upload_id = filter_upload_id
        AND (filter_slide_number IS NULL OR sc.slide_number = filter_slide_number)
    ORDER BY sc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Full-text search RPC function
CREATE OR REPLACE FUNCTION search_slide_chunks_text(
    query_text TEXT,
    filter_upload_id TEXT,
    match_count INTEGER DEFAULT 10,
    filter_slide_number INTEGER DEFAULT NULL
)
RETURNS TABLE (
    id TEXT,
    upload_id TEXT,
    slide_number INTEGER,
    chunk_index INTEGER,
    title TEXT,
    content_type TEXT,
    content TEXT,
    rank FLOAT
)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        sc.id, sc.upload_id, sc.slide_number, sc.chunk_index,
        sc.title, sc.content_type, sc.content,
        ts_rank_cd(sc.search_vector, websearch_to_tsquery('english', query_text))::FLOAT AS rank
    FROM slide_chunks sc
    WHERE sc.upload_id = filter_upload_id
        AND sc.search_vector @@ websearch_to_tsquery('english', query_text)
        AND (filter_slide_number IS NULL OR sc.slide_number = filter_slide_number)
    ORDER BY rank DESC
    LIMIT match_count;
END;
$$;
