"""Tests for token authentication functionality."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials

from agent_memory_server.auth import (
    TokenInfo,
    generate_token,
    get_current_user,
    hash_token,
    verify_auth_config,
    verify_token,
    verify_token_hash,
)
from agent_memory_server.config import settings
from agent_memory_server.utils.keys import Keys


class AsyncIterator:
    """Helper class for creating async iterators in tests."""

    def __init__(self, items):
        self.items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.items)
        except StopIteration:
            raise StopAsyncIteration from None


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    original_settings = {
        "disable_auth": settings.disable_auth,
        "auth_mode": settings.auth_mode,
        "token_auth_enabled": settings.token_auth_enabled,
    }

    yield settings

    # Restore original settings
    for key, value in original_settings.items():
        setattr(settings, key, value)


@pytest.fixture
def mock_redis():
    """Mock Redis connection."""
    return AsyncMock()


@pytest.fixture
def sample_token_info():
    """Sample token info for testing."""
    return TokenInfo(
        description="Test token",
        created_at=datetime.now(UTC),
        expires_at=datetime.now(UTC) + timedelta(days=30),
        token_hash="$2b$12$test_hash_here",
    )


class TestTokenGeneration:
    """Test token generation and hashing."""

    def test_generate_token(self):
        """Test token generation."""
        token = generate_token()
        assert isinstance(token, str)
        assert len(token) > 0

        # Tokens should be unique
        token2 = generate_token()
        assert token != token2

    def test_hash_token(self):
        """Test token hashing."""
        token = "test_token_123"
        token_hash = hash_token(token)

        assert isinstance(token_hash, str)
        assert len(token_hash) > 0
        assert token_hash != token
        assert token_hash.startswith("$2b$")

    def test_verify_token_hash(self):
        """Test token hash verification."""
        token = "test_token_123"
        token_hash = hash_token(token)

        # Correct token should verify
        assert verify_token_hash(token, token_hash) is True

        # Wrong token should not verify
        assert verify_token_hash("wrong_token", token_hash) is False

        # Invalid hash should return False
        assert verify_token_hash(token, "invalid_hash") is False


class TestTokenVerification:
    """Test token verification functionality."""

    @pytest.mark.asyncio
    async def test_verify_token_success(self, mock_redis, sample_token_info):
        """Test successful token verification."""
        token = "test_token_123"
        token_hash = hash_token(token)
        sample_token_info.token_hash = token_hash

        mock_redis.scan_iter = Mock(
            return_value=AsyncIterator([Keys.auth_token_key(token_hash)])
        )
        mock_redis.get.return_value = sample_token_info.model_dump_json()

        with patch("agent_memory_server.auth.get_redis_conn", return_value=mock_redis):
            user_info = await verify_token(token)

            assert user_info.sub == "token-user"
            assert user_info.aud == "token-auth"
            assert user_info.scope == "admin"
            assert "admin" in user_info.roles

    @pytest.mark.asyncio
    async def test_verify_token_expired(self, mock_redis, sample_token_info):
        """Test verification of expired token."""
        token = "test_token_123"
        token_hash = hash_token(token)
        sample_token_info.token_hash = token_hash
        sample_token_info.expires_at = datetime.now(UTC) - timedelta(days=1)  # Expired

        mock_redis.scan_iter = Mock(
            return_value=AsyncIterator([Keys.auth_token_key(token_hash)])
        )
        mock_redis.get.return_value = sample_token_info.model_dump_json()

        with patch("agent_memory_server.auth.get_redis_conn", return_value=mock_redis):
            with pytest.raises(HTTPException) as exc_info:
                await verify_token(token)

            assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
            assert "expired" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_verify_token_not_found(self, mock_redis):
        """Test verification of non-existent token."""
        token = "nonexistent_token"

        # Mock Redis responses - no tokens found
        mock_redis.scan_iter = Mock(return_value=AsyncIterator([]))

        with patch("agent_memory_server.auth.get_redis_conn", return_value=mock_redis):
            with pytest.raises(HTTPException) as exc_info:
                await verify_token(token)

            assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
            assert "Invalid token" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_verify_token_wrong_token(self, mock_redis, sample_token_info):
        """Test verification with wrong token."""
        correct_token = "test_token_123"
        wrong_token = "wrong_token_456"
        token_hash = hash_token(correct_token)
        sample_token_info.token_hash = token_hash

        mock_redis.scan_iter = Mock(
            return_value=AsyncIterator([Keys.auth_token_key(token_hash)])
        )
        mock_redis.get.return_value = sample_token_info.model_dump_json()

        with patch("agent_memory_server.auth.get_redis_conn", return_value=mock_redis):
            with pytest.raises(HTTPException) as exc_info:
                await verify_token(wrong_token)

            assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
            assert "Invalid token" in exc_info.value.detail


class TestGetCurrentUser:
    """Test get_current_user with token authentication."""

    def test_get_current_user_disabled_auth(self, mock_settings):
        """Test get_current_user with disabled authentication."""
        mock_settings.disable_auth = True
        mock_settings.auth_mode = "disabled"

        user_info = get_current_user(None)

        assert user_info.sub == "local-dev-user"
        assert user_info.aud == "local-dev"

    def test_get_current_user_missing_credentials(self, mock_settings):
        """Test get_current_user with missing credentials."""
        mock_settings.disable_auth = False
        mock_settings.auth_mode = "token"

        with pytest.raises(HTTPException) as exc_info:
            get_current_user(None)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Missing authorization header" in exc_info.value.detail

    def test_get_current_user_missing_token(self, mock_settings):
        """Test get_current_user with missing token."""
        mock_settings.disable_auth = False
        mock_settings.auth_mode = "token"

        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="")

        with pytest.raises(HTTPException) as exc_info:
            get_current_user(credentials)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Missing bearer token" in exc_info.value.detail

    @patch("agent_memory_server.auth.verify_token")
    def test_get_current_user_token_auth(self, mock_verify_token, mock_settings):
        """Test get_current_user with token authentication."""
        mock_settings.disable_auth = False
        mock_settings.auth_mode = "token"

        # Mock verify_token to return a user
        mock_user = Mock()
        mock_user.sub = "token-user"

        # Mock asyncio.run to return the user directly
        with patch("asyncio.run", return_value=mock_user):
            credentials = HTTPAuthorizationCredentials(
                scheme="Bearer", credentials="test_token"
            )

            user_info = get_current_user(credentials)

            assert user_info.sub == "token-user"


class TestAuthConfig:
    """Test authentication configuration validation."""

    def test_verify_auth_config_disabled(self, mock_settings):
        """Test auth config verification when disabled."""
        mock_settings.disable_auth = True
        mock_settings.auth_mode = "disabled"

        # Should not raise any exception
        verify_auth_config()

    def test_verify_auth_config_token_mode(self, mock_settings):
        """Test auth config verification for token mode."""
        mock_settings.disable_auth = False
        mock_settings.auth_mode = "token"

        # Should not raise any exception
        verify_auth_config()

    def test_verify_auth_config_token_enabled(self, mock_settings):
        """Test auth config verification when token_auth_enabled is True."""
        mock_settings.disable_auth = False
        mock_settings.auth_mode = "disabled"
        mock_settings.token_auth_enabled = True

        # Should not raise any exception
        verify_auth_config()


class TestTokenInfo:
    """Test TokenInfo model."""

    def test_token_info_creation(self):
        """Test TokenInfo model creation."""
        now = datetime.now(UTC)
        expires = now + timedelta(days=30)

        token_info = TokenInfo(
            description="Test token",
            created_at=now,
            expires_at=expires,
            token_hash="test_hash",
        )

        assert token_info.description == "Test token"
        assert token_info.created_at == now
        assert token_info.expires_at == expires
        assert token_info.token_hash == "test_hash"

    def test_token_info_json_serialization(self):
        """Test TokenInfo JSON serialization."""
        now = datetime.now(UTC)
        expires = now + timedelta(days=30)

        token_info = TokenInfo(
            description="Test token",
            created_at=now,
            expires_at=expires,
            token_hash="test_hash",
        )

        json_str = token_info.model_dump_json()
        assert isinstance(json_str, str)

        # Verify it can be parsed back
        parsed = TokenInfo.model_validate_json(json_str)
        assert parsed.description == token_info.description
        assert parsed.token_hash == token_info.token_hash

    def test_token_info_no_expiration(self):
        """Test TokenInfo without expiration."""
        now = datetime.now(UTC)

        token_info = TokenInfo(
            description="Permanent token",
            created_at=now,
            expires_at=None,
            token_hash="test_hash",
        )

        assert token_info.expires_at is None


class TestKeys:
    """Test Redis key generation for tokens."""

    def test_auth_token_key(self):
        """Test auth token key generation."""
        token_hash = "test_hash_123"
        key = Keys.auth_token_key(token_hash)

        assert key == f"auth_token:{token_hash}"

    def test_auth_tokens_list_key(self):
        """Test auth tokens list key generation."""
        key = Keys.auth_tokens_list_key()

        assert key == "auth_tokens:list"
