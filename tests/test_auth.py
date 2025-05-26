import json
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient
from jose import jwt

from agent_memory_server.auth import clear_jwks_cache, get_current_user, verify_jwt_token
from agent_memory_server.config import Settings


# Mock JWKS data for testing
MOCK_JWKS = {
    "keys": [
        {
            "kty": "RSA",
            "kid": "test-key-id",
            "use": "sig",
            "alg": "RS256",
            "n": "mock_n_value",
            "e": "AQAB"
        }
    ]
}

# Test RSA key pair for JWT signing (for testing only)
TEST_PRIVATE_KEY = """-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEAwJKGYqY9NX2K2xZ2w8vXF3q5bQjVo8A0w7zK9X3j2q1X8cZX
mock_private_key_data_for_testing_only
-----END RSA PRIVATE KEY-----"""

TEST_PUBLIC_KEY = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAwJKGYqY9NX2K2xZ2w8vX
mock_public_key_data_for_testing_only
-----END PUBLIC KEY-----"""


@pytest.fixture
def auth_settings():
    """Create test settings with OAuth2 configuration"""
    return Settings(
        oauth2_issuer_url="https://test-issuer.example.com",
        oauth2_audience="test-audience",
        disable_auth=False,
        redis_url="redis://localhost:6379",
    )


@pytest.fixture
def disabled_auth_settings():
    """Create test settings with authentication disabled"""
    return Settings(
        disable_auth=True,
        redis_url="redis://localhost:6379",
    )


@pytest.fixture
def valid_jwt_token():
    """Create a valid JWT token for testing"""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": "test-user-123",
        "iss": "https://test-issuer.example.com", 
        "aud": "test-audience",
        "exp": now + timedelta(hours=1),
        "iat": now,
        "scope": "read write"
    }
    
    # Create a simple token (won't verify with real JWKS but good for basic tests)
    return jwt.encode(payload, "test-secret", algorithm="HS256")


@pytest.fixture
def expired_jwt_token():
    """Create an expired JWT token for testing"""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": "test-user-123",
        "iss": "https://test-issuer.example.com",
        "aud": "test-audience", 
        "exp": now - timedelta(hours=1),  # Expired
        "iat": now - timedelta(hours=2),
    }
    
    return jwt.encode(payload, "test-secret", algorithm="HS256")


class TestAuthDisabled:
    """Test authentication when DISABLE_AUTH=True"""
    
    @pytest.mark.asyncio
    async def test_get_current_user_disabled_auth(self, disabled_auth_settings):
        """Test that authentication is bypassed when DISABLE_AUTH=True"""
        with patch("agent_memory_server.auth.settings", disabled_auth_settings):
            user = await get_current_user(credentials=None)
            
        assert user["sub"] == "local-dev-user"
        assert user["roles"] == ["admin"]
        assert user["iss"] == "local-dev"


class TestJWTValidation:
    """Test JWT token validation"""
    
    @pytest.mark.asyncio 
    async def test_verify_jwt_missing_kid(self, auth_settings):
        """Test JWT validation fails when kid is missing from header"""
        # Create token without kid in header
        payload = {"sub": "test-user", "iss": "test-issuer", "aud": "test-audience"}
        token_no_kid = jwt.encode(payload, "secret", algorithm="HS256", headers={})
        
        with patch("agent_memory_server.auth.settings", auth_settings):
            with pytest.raises(Exception) as exc_info:
                await verify_jwt_token(token_no_kid)
            
        assert "kid" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_verify_jwt_invalid_signature(self, auth_settings):
        """Test JWT validation fails with invalid signature"""
        with patch("agent_memory_server.auth.settings", auth_settings):
            # Mock JWKS response
            with patch("agent_memory_server.auth.get_jwks_keys") as mock_jwks:
                mock_jwks.return_value = MOCK_JWKS
                
                # Create token with different secret than expected
                payload = {
                    "sub": "test-user",
                    "iss": "https://test-issuer.example.com",
                    "aud": "test-audience"
                }
                headers = {"kid": "test-key-id"}
                invalid_token = jwt.encode(payload, "wrong-secret", algorithm="HS256", headers=headers)
                
                with pytest.raises(Exception):
                    await verify_jwt_token(invalid_token)


class TestJWKSCaching:
    """Test JWKS key caching functionality"""
    
    @pytest.mark.asyncio
    async def test_jwks_cache_clearing(self):
        """Test that JWKS cache can be cleared"""
        # This is a simple test since we can't easily test the full caching mechanism
        clear_jwks_cache()
        # No exception should be raised
        assert True
    
    @pytest.mark.asyncio
    async def test_get_jwks_keys_network_error(self, auth_settings):
        """Test JWKS fetching handles network errors gracefully"""
        with patch("agent_memory_server.auth.settings", auth_settings):
            with patch("httpx.AsyncClient.get") as mock_get:
                mock_get.side_effect = Exception("Network error")
                
                with pytest.raises(Exception) as exc_info:
                    from agent_memory_server.auth import get_jwks_keys
                    await get_jwks_keys()
                
                assert "Unable to fetch JWKS keys" in str(exc_info.value)


class TestFastAPIIntegration:
    """Test OAuth2 authentication integration with FastAPI endpoints"""
    
    def test_protected_endpoint_without_token(self, client):
        """Test that protected endpoints return 401 without token"""
        # Enable auth for this test
        with patch("agent_memory_server.auth.settings") as mock_settings:
            mock_settings.disable_auth = False
            mock_settings.oauth2_issuer_url = "https://test-issuer.example.com"
            mock_settings.oauth2_audience = "test-audience"
            
            response = client.get("/sessions/")
            
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_protected_endpoint_with_auth_disabled(self, client):
        """Test that protected endpoints work when auth is disabled"""
        with patch("agent_memory_server.auth.settings") as mock_settings:
            mock_settings.disable_auth = True
            
            # Mock Redis and other dependencies
            with patch("agent_memory_server.utils.redis.get_redis_conn") as mock_redis:
                mock_redis.return_value = AsyncMock()
                with patch("agent_memory_server.messages.list_sessions") as mock_list:
                    mock_list.return_value = (0, [])
                    
                    response = client.get("/sessions/")
                    
        assert response.status_code == status.HTTP_200_OK
    
    def test_health_endpoint_always_accessible(self, client):
        """Test that /health endpoint is always accessible"""
        # Health endpoint should work regardless of auth settings
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        
        # Even with auth enabled, health should be accessible
        with patch("agent_memory_server.auth.settings") as mock_settings:
            mock_settings.disable_auth = False
            mock_settings.oauth2_issuer_url = "https://test-issuer.example.com"
            
            response = client.get("/health")
            assert response.status_code == status.HTTP_200_OK


class TestConfigurationValidation:
    """Test OAuth2 configuration validation"""
    
    @pytest.mark.asyncio
    async def test_missing_issuer_url_configuration(self):
        """Test that missing issuer URL is handled properly"""
        with patch("agent_memory_server.auth.settings") as mock_settings:
            mock_settings.oauth2_issuer_url = None
            mock_settings.disable_auth = False
            
            with pytest.raises(Exception) as exc_info:
                from agent_memory_server.auth import get_jwks_keys
                await get_jwks_keys()
            
            assert "OAuth2 issuer URL not configured" in str(exc_info.value)
    
    def test_oauth2_settings_from_env(self):
        """Test that OAuth2 settings can be loaded from environment variables"""
        test_env = {
            "OAUTH2_ISSUER_URL": "https://auth.example.com",
            "OAUTH2_AUDIENCE": "my-api",
            "OAUTH2_JWKS_URL": "https://auth.example.com/.well-known/jwks.json",
            "DISABLE_AUTH": "false"
        }
        
        with patch.dict(os.environ, test_env):
            settings = Settings()
            
            assert settings.oauth2_issuer_url == "https://auth.example.com"
            assert settings.oauth2_audience == "my-api"
            assert settings.oauth2_jwks_url == "https://auth.example.com/.well-known/jwks.json"
            assert settings.disable_auth is False


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    @pytest.mark.asyncio
    async def test_malformed_jwt_token(self, auth_settings):
        """Test handling of malformed JWT tokens"""
        with patch("agent_memory_server.auth.settings", auth_settings):
            malformed_token = "not.a.valid.jwt.token"
            
            with pytest.raises(Exception):
                await verify_jwt_token(malformed_token)
    
    @pytest.mark.asyncio
    async def test_jwt_without_required_claims(self, auth_settings):
        """Test JWT validation fails when required claims are missing"""
        with patch("agent_memory_server.auth.settings", auth_settings):
            # Create token missing required claims
            payload = {"sub": "test-user"}  # Missing iss, aud, etc.
            headers = {"kid": "test-key-id"}
            token = jwt.encode(payload, "secret", algorithm="HS256", headers=headers)
            
            with patch("agent_memory_server.auth.get_jwks_keys") as mock_jwks:
                mock_jwks.return_value = MOCK_JWKS
                
                with pytest.raises(Exception):
                    await verify_jwt_token(token)


# Test data for documentation examples
class TestDocumentationExamples:
    """Test examples that would be shown in documentation"""
    
    def test_disable_auth_environment_variable(self):
        """Test the DISABLE_AUTH environment variable as documented"""
        # Test that DISABLE_AUTH=true works
        with patch.dict(os.environ, {"DISABLE_AUTH": "true"}):
            settings = Settings()
            assert settings.disable_auth is True
        
        # Test that DISABLE_AUTH=false works  
        with patch.dict(os.environ, {"DISABLE_AUTH": "false"}):
            settings = Settings()
            assert settings.disable_auth is False
        
        # Test default value
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.disable_auth is False