"""Tests for FIPS compliance of cryptographic operations."""

import hashlib
import ssl
from unittest.mock import AsyncMock

import pytest

from agent_memory_server.auth import hash_token, verify_token_hash
from agent_memory_server.config import settings
from agent_memory_server.fips import get_fips_diagnostics
from agent_memory_server.utils.recency import (
    generate_memory_hash,
    generate_memory_hash_from_fields,
)


@pytest.fixture
def _save_settings():
    """Save and restore settings modified during tests."""
    original = {
        "token_hash_algorithm": settings.token_hash_algorithm,
        "token_hash_secret": settings.token_hash_secret,
        "token_hash_iterations": settings.token_hash_iterations,
        "redis_url": settings.redis_url,
        "redis_ssl_ca_certs": settings.redis_ssl_ca_certs,
        "redis_ssl_certfile": settings.redis_ssl_certfile,
        "redis_ssl_keyfile": settings.redis_ssl_keyfile,
        "redis_ssl_cert_reqs": settings.redis_ssl_cert_reqs,
        "redis_ssl_min_version": settings.redis_ssl_min_version,
    }
    yield
    for key, value in original.items():
        setattr(settings, key, value)


class TestFIPSTokenHashing:
    """Test FIPS-approved token hashing algorithms."""

    def test_hmac_sha256_hash_and_verify(self, _save_settings):
        settings.token_hash_algorithm = "hmac-sha256"
        settings.token_hash_secret = "test-secret-key"

        token = "my-secret-token"
        hashed = hash_token(token)

        assert hashed.startswith("hmac$")
        assert verify_token_hash(token, hashed) is True
        assert verify_token_hash("wrong-token", hashed) is False

    def test_hmac_sha256_deterministic(self, _save_settings):
        settings.token_hash_algorithm = "hmac-sha256"
        settings.token_hash_secret = "test-secret-key"

        token = "my-secret-token"
        hash1 = hash_token(token)
        hash2 = hash_token(token)
        assert hash1 == hash2

    def test_pbkdf2_sha256_hash_and_verify(self, _save_settings):
        settings.token_hash_algorithm = "pbkdf2-sha256"
        settings.token_hash_iterations = 1000  # Low for test speed

        token = "my-secret-token"
        hashed = hash_token(token)

        assert hashed.startswith("pbkdf2$")
        parts = hashed.split("$")
        assert len(parts) == 3
        assert verify_token_hash(token, hashed) is True
        assert verify_token_hash("wrong-token", hashed) is False

    def test_pbkdf2_sha256_unique_salts(self, _save_settings):
        settings.token_hash_algorithm = "pbkdf2-sha256"
        settings.token_hash_iterations = 1000

        token = "my-secret-token"
        hash1 = hash_token(token)
        hash2 = hash_token(token)
        # Different salts produce different hashes
        assert hash1 != hash2
        # But both verify
        assert verify_token_hash(token, hash1) is True
        assert verify_token_hash(token, hash2) is True

    def test_bcrypt_hash_and_verify(self, _save_settings):
        settings.token_hash_algorithm = "bcrypt"

        token = "my-secret-token"
        hashed = hash_token(token)

        assert hashed.startswith("$2b$")
        assert verify_token_hash(token, hashed) is True
        assert verify_token_hash("wrong-token", hashed) is False

    def test_backward_compat_bcrypt_detection(self, _save_settings):
        """Verify that existing bcrypt hashes are auto-detected and verified
        regardless of the current token_hash_algorithm setting."""
        settings.token_hash_algorithm = "bcrypt"
        token = "my-secret-token"
        bcrypt_hash = hash_token(token)

        # Switch to hmac-sha256 (new default)
        settings.token_hash_algorithm = "hmac-sha256"
        settings.token_hash_secret = "test-secret"

        # Old bcrypt hash should still verify
        assert verify_token_hash(token, bcrypt_hash) is True

    def test_verify_unknown_prefix_returns_false(self, _save_settings):
        assert verify_token_hash("token", "unknown$prefix$hash") is False

    def test_unsupported_algorithm_raises(self, _save_settings):
        settings.token_hash_algorithm = "unsupported"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unsupported token hash algorithm"):
            hash_token("token")

    def test_verify_malformed_hash_returns_false(self, _save_settings):
        """Malformed hashes should not raise, just return False."""
        assert verify_token_hash("token", "pbkdf2$not-hex$data") is False
        assert verify_token_hash("token", "") is False


class TestFIPSHashlib:
    """Test that hashlib usage is FIPS-compatible."""

    def test_memory_hash_uses_sha256(self):
        """Verify memory hashing produces valid SHA-256 hex digests."""
        result = generate_memory_hash_from_fields(
            text="test",
            user_id="user1",
            session_id="sess1",
            namespace="ns1",
            memory_type="message",
        )
        # SHA-256 produces 64 hex characters
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_memory_hash_from_record(self):
        """Verify generate_memory_hash works with MemoryRecord objects."""
        from agent_memory_server.models import MemoryRecord

        record = MemoryRecord(
            id="test-id",
            text="test text",
            user_id="user1",
            session_id="sess1",
            namespace="ns1",
        )
        result = generate_memory_hash(record)
        assert len(result) == 64

    def test_sha256_usedforsecurity_false(self):
        """Verify that SHA-256 with usedforsecurity=False works correctly."""
        # This should work even on strict FIPS systems
        h = hashlib.sha256(b"test", usedforsecurity=False)
        assert len(h.hexdigest()) == 64


class TestFIPSDiagnostics:
    """Test FIPS diagnostics reporting."""

    def test_diagnostics_structure(self, _save_settings):
        diagnostics = get_fips_diagnostics()

        assert "openssl_version" in diagnostics
        assert "kernel_fips_enabled" in diagnostics
        assert "md5_blocked" in diagnostics
        assert "token_hash_algorithm" in diagnostics
        assert "redis_tls_enabled" in diagnostics
        assert "fips_capable" in diagnostics

    def test_diagnostics_openssl_version(self, _save_settings):
        diagnostics = get_fips_diagnostics()
        assert diagnostics["openssl_version"] == ssl.OPENSSL_VERSION

    def test_diagnostics_token_algorithm(self, _save_settings):
        settings.token_hash_algorithm = "hmac-sha256"
        diagnostics = get_fips_diagnostics()
        assert diagnostics["token_hash_algorithm"] == "hmac-sha256"

    def test_diagnostics_redis_tls(self, _save_settings):
        settings.redis_url = "redis://localhost:6379"
        diagnostics = get_fips_diagnostics()
        assert diagnostics["redis_tls_enabled"] is False

        settings.redis_url = "rediss://localhost:6380"
        diagnostics = get_fips_diagnostics()
        assert diagnostics["redis_tls_enabled"] is True

    def test_diagnostics_fips_capable_when_bcrypt(self, _save_settings):
        settings.token_hash_algorithm = "bcrypt"
        diagnostics = get_fips_diagnostics()
        # bcrypt means not fips-capable regardless of md5 status
        assert diagnostics["fips_capable"] is False

    @pytest.mark.asyncio
    async def test_fips_endpoint(self, _save_settings):
        from agent_memory_server.healthcheck import get_fips_status

        user = AsyncMock()
        result = await get_fips_status(user=user)
        assert "openssl_version" in result
        assert "fips_capable" in result


class TestRedisTLSConfig:
    """Test Redis TLS configuration."""

    def test_tls_kwargs_with_rediss_url(self, _save_settings):
        """TLS kwargs should be applied when using rediss:// URL."""
        from agent_memory_server.utils.redis import build_redis_tls_kwargs

        settings.redis_ssl_ca_certs = "/path/to/ca.pem"
        settings.redis_ssl_certfile = "/path/to/cert.pem"
        settings.redis_ssl_keyfile = "/path/to/key.pem"
        settings.redis_ssl_cert_reqs = "required"
        settings.redis_ssl_min_version = "TLSv1_2"

        kwargs = build_redis_tls_kwargs("rediss://localhost:6380")
        assert kwargs["ssl_ca_certs"] == "/path/to/ca.pem"
        assert kwargs["ssl_certfile"] == "/path/to/cert.pem"
        assert kwargs["ssl_keyfile"] == "/path/to/key.pem"
        assert kwargs["ssl_cert_reqs"] == "required"
        assert kwargs["ssl_min_version"] == ssl.TLSVersion.TLSv1_2

    def test_no_tls_kwargs_with_redis_url(self, _save_settings):
        """TLS kwargs should NOT be applied when using redis:// URL."""
        from agent_memory_server.utils.redis import build_redis_tls_kwargs

        settings.redis_ssl_ca_certs = None
        settings.redis_ssl_certfile = None
        settings.redis_ssl_keyfile = None

        kwargs = build_redis_tls_kwargs("redis://localhost:6379")
        assert "ssl_ca_certs" not in kwargs
        assert "ssl_certfile" not in kwargs
        assert "ssl_min_version" not in kwargs

    def test_tls_kwargs_with_ca_certs_override(self, _save_settings):
        """TLS should activate when redis_ssl_ca_certs is set, even with redis:// URL."""
        from agent_memory_server.utils.redis import build_redis_tls_kwargs

        settings.redis_ssl_ca_certs = "/path/to/ca.pem"
        settings.redis_ssl_certfile = None
        settings.redis_ssl_keyfile = None

        kwargs = build_redis_tls_kwargs("redis://localhost:6379")
        assert kwargs["ssl_ca_certs"] == "/path/to/ca.pem"
        assert "ssl_certfile" not in kwargs
        assert "ssl_keyfile" not in kwargs
        # cert_reqs and min_version are always set when TLS is active
        assert kwargs["ssl_cert_reqs"] == "required"
        assert kwargs["ssl_min_version"] == ssl.TLSVersion.TLSv1_2

    def test_ssl_min_version_tls13(self, _save_settings):
        """Test TLSv1_3 min version setting."""
        from agent_memory_server.utils.redis import build_redis_tls_kwargs

        settings.redis_ssl_ca_certs = "/path/to/ca.pem"
        settings.redis_ssl_min_version = "TLSv1_3"

        kwargs = build_redis_tls_kwargs("rediss://localhost:6380")
        assert kwargs["ssl_min_version"] == ssl.TLSVersion.TLSv1_3

    def test_ssl_min_version_invalid_fallback(self, _save_settings):
        """Invalid version name falls back to TLSv1_2."""
        from agent_memory_server.utils.redis import build_redis_tls_kwargs

        settings.redis_ssl_ca_certs = "/path/to/ca.pem"
        settings.redis_ssl_min_version = "INVALID"

        kwargs = build_redis_tls_kwargs("rediss://localhost:6380")
        assert kwargs["ssl_min_version"] == ssl.TLSVersion.TLSv1_2

    def test_existing_kwargs_not_overwritten(self, _save_settings):
        """Caller-supplied kwargs should not be overwritten by settings."""
        from agent_memory_server.utils.redis import build_redis_tls_kwargs

        settings.redis_ssl_ca_certs = "/path/to/ca.pem"
        settings.redis_ssl_cert_reqs = "required"

        kwargs = build_redis_tls_kwargs(
            "rediss://localhost:6380",
            ssl_cert_reqs="none",
        )
        # Caller's value should take precedence via setdefault
        assert kwargs["ssl_cert_reqs"] == "none"
