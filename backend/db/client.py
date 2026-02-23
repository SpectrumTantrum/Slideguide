"""
Supabase client singleton for backend operations.

Uses the service role key since there's no auth — all operations
are unrestricted. The client is sync (supabase-py v2 default).
"""
from __future__ import annotations

from supabase import Client, create_client

from backend.config import settings
from backend.monitoring.logger import get_logger

logger = get_logger(__name__)

_client: Client | None = None


def get_supabase() -> Client:
    """Get or create the Supabase client singleton."""
    global _client
    if _client is None:
        _client = create_client(
            settings.supabase_url,
            settings.supabase_service_role_key,
        )
        logger.info("supabase_client_initialized", url=settings.supabase_url)
    return _client


def reset_client() -> None:
    """Reset the client (for testing)."""
    global _client
    _client = None
