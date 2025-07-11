"""Redis key utilities."""

import logging

from agent_memory_server.config import settings


logger = logging.getLogger(__name__)


class Keys:
    """Redis key utilities."""

    @staticmethod
    def context_key(session_id: str, namespace: str | None = None) -> str:
        """Get the context key for a session."""
        return (
            f"context:{namespace}:{session_id}"
            if namespace
            else f"context:{session_id}"
        )

    @staticmethod
    def token_count_key(session_id: str, namespace: str | None = None) -> str:
        """Get the token count key for a session."""
        return (
            f"tokens:{namespace}:{session_id}" if namespace else f"tokens:{session_id}"
        )

    @staticmethod
    def messages_key(session_id: str, namespace: str | None = None) -> str:
        """Get the messages key for a session."""
        return (
            f"messages:{namespace}:{session_id}"
            if namespace
            else f"messages:{session_id}"
        )

    @staticmethod
    def sessions_key(namespace: str | None = None) -> str:
        """Get the sessions key for a namespace."""
        return f"sessions:{namespace}" if namespace else "sessions"

    @staticmethod
    def memory_key(id: str) -> str:
        """Get the memory key for an ID."""
        return f"{settings.redisvl_index_prefix}:{id}"

    @staticmethod
    def metadata_key(session_id: str, namespace: str | None = None) -> str:
        """Get the metadata key for a session."""
        return (
            f"metadata:{namespace}:{session_id}"
            if namespace
            else f"metadata:{session_id}"
        )

    @staticmethod
    def working_memory_key(
        session_id: str, user_id: str | None = None, namespace: str | None = None
    ) -> str:
        """Get the working memory key for a session."""
        # Build key components, filtering out None values
        key_parts = ["working_memory"]

        if namespace:
            key_parts.append(namespace)

        if user_id:
            key_parts.append(user_id)

        key_parts.append(session_id)

        return ":".join(key_parts)

    @staticmethod
    def search_index_name() -> str:
        """Return the name of the search index."""
        return settings.redisvl_index_name

    @staticmethod
    def auth_token_key(token_hash: str) -> str:
        """Get the auth token key for a hashed token."""
        return f"auth_token:{token_hash}"

    @staticmethod
    def auth_tokens_list_key() -> str:
        """Get the key for the list of all auth tokens."""
        return "auth_tokens:list"
