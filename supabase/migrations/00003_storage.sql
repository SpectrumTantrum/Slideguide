-- Create storage bucket for slide file uploads
INSERT INTO storage.buckets (id, name, public)
VALUES ('slides', 'slides', false)
ON CONFLICT (id) DO NOTHING;
