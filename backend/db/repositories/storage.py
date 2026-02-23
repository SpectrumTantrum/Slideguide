"""Supabase Storage repository for slide file uploads."""

from __future__ import annotations

from supabase import Client

from backend.monitoring.logger import get_logger

logger = get_logger(__name__)

BUCKET = "slides"


class StorageRepository:
    """Upload, download, and manage files in Supabase Storage."""

    def __init__(self, client: Client) -> None:
        self.client = client

    def upload_file(
        self,
        upload_id: str,
        filename: str,
        content: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload file to Supabase Storage. Returns the storage path."""
        path = f"{upload_id}/{filename}"
        self.client.storage.from_(BUCKET).upload(
            path, content, {"content-type": content_type}
        )
        logger.info("file_uploaded", path=path, size=len(content))
        return path

    def download_file(self, path: str) -> bytes:
        """Download file from Supabase Storage."""
        return self.client.storage.from_(BUCKET).download(path)

    def get_signed_url(self, path: str, expires_in: int = 3600) -> str:
        """Get a signed URL for temporary access."""
        result = self.client.storage.from_(BUCKET).create_signed_url(path, expires_in)
        return result["signedURL"]

    def delete_file(self, path: str) -> None:
        """Delete a file from storage."""
        self.client.storage.from_(BUCKET).remove([path])
