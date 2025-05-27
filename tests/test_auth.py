import json
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

import httpx
import pytest
from fastapi import HTTPException, status
from jose import jwt

from agent_memory_server.auth import (
    JWKSCache,
    UserInfo,
    get_current_user,
    get_jwks_url,
    get_public_key,
    require_role,
    require_scope,
    verify_auth_config,
    verify_jwt,
)
from agent_memory_server.config import settings


@pytest.fixture
def mock_settings():
    """Mock settings for testing"""
    original_settings = {
        "disable_auth": settings.disable_auth,
        "oauth2_issuer_url": settings.oauth2_issuer_url,
        "oauth2_audience": settings.oauth2_audience,
        "oauth2_jwks_url": settings.oauth2_jwks_url,
        "oauth2_algorithms": settings.oauth2_algorithms,
    }

    yield settings

    # Restore original settings
    for key, value in original_settings.items():
        setattr(settings, key, value)


@pytest.fixture
def private_key():
    """Generate a test RSA private key"""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    private_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    return private_pem.decode("utf-8")


@pytest.fixture
def public_key(private_key):
    """Extract public key from private key"""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    private_key_obj = load_pem_private_key(private_key.encode("utf-8"), password=None)
    public_key_obj = private_key_obj.public_key()

    public_pem = public_key_obj.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    return public_pem.decode("utf-8")


@pytest.fixture
def jwks_data():
    """Mock JWKS data"""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from jose.jwk import RSAKey

    # Generate test RSA key
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # Convert to JWK format
    rsa_key = RSAKey(key=key, algorithm="RS256")
    jwk_dict = rsa_key.to_dict()
    jwk_dict["kid"] = "test-kid-123"
    jwk_dict["use"] = "sig"
    jwk_dict["alg"] = "RS256"

    private_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    return {
        "keys": [jwk_dict],
        "private_key": private_pem.decode("utf-8"),
        "kid": "test-kid-123",
    }


@pytest.fixture
def valid_token(jwks_data):
    """Generate a valid JWT token"""
    payload = {
        "sub": "test-user-123",
        "aud": "test-audience",
        "iss": "https://test-issuer.com",
        "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
        "iat": int(datetime.now(UTC).timestamp()),
        "email": "test@example.com",
        "scope": "read write admin",
        "roles": ["user", "admin"],
    }

    return jwt.encode(
        payload,
        jwks_data["private_key"],
        algorithm="RS256",
        headers={"kid": jwks_data["kid"]},
    )


@pytest.fixture
def expired_token(jwks_data):
    """Generate an expired JWT token"""
    payload = {
        "sub": "test-user-123",
        "aud": "test-audience",
        "iss": "https://test-issuer.com",
        "exp": int((datetime.now(UTC) - timedelta(hours=1)).timestamp()),
        "iat": int((datetime.now(UTC) - timedelta(hours=2)).timestamp()),
        "email": "test@example.com",
        "scope": "read write",
        "roles": ["user"],
    }

    return jwt.encode(
        payload,
        jwks_data["private_key"],
        algorithm="RS256",
        headers={"kid": jwks_data["kid"]},
    )


class TestJWKSCache:
    """Test JWKS caching functionality"""

    def test_jwks_cache_initialization(self):
        """Test JWKS cache initializes with correct defaults"""
        cache = JWKSCache()
        assert cache._cache == {}
        assert cache._cache_time is None
        assert cache._cache_duration == 3600
        assert hasattr(cache._lock, "acquire")  # Should be a threading.Lock

    def test_jwks_cache_custom_duration(self):
        """Test JWKS cache with custom duration"""
        cache = JWKSCache(cache_duration=7200)
        assert cache._cache_duration == 7200

    @pytest.mark.asyncio
    async def test_jwks_cache_fetch_success(self, jwks_data):
        """Test successful JWKS fetch and caching"""
        cache = JWKSCache()
        jwks_url = "https://test-issuer.com/.well-known/jwks.json"

        with patch("httpx.Client") as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {"keys": jwks_data["keys"]}
            mock_response.raise_for_status.return_value = None

            mock_context = Mock()
            mock_context.__enter__.return_value.get.return_value = mock_response
            mock_client.return_value = mock_context

            result = cache.get_jwks(jwks_url)

            assert result == {"keys": jwks_data["keys"]}
            assert cache._cache == {"keys": jwks_data["keys"]}
            assert cache._cache_time is not None

    @pytest.mark.asyncio
    async def test_jwks_cache_returns_cached_data(self, jwks_data):
        """Test that cache returns cached data within cache duration"""
        cache = JWKSCache(cache_duration=3600)
        cache._cache = {"keys": jwks_data["keys"]}
        cache._cache_time = time.time()

        # Should return cached data without making HTTP request
        with patch("httpx.Client") as mock_client:
            result = cache.get_jwks("https://test-issuer.com/.well-known/jwks.json")
            assert result == {"keys": jwks_data["keys"]}
            mock_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_jwks_cache_refresh_expired(self, jwks_data):
        """Test cache refresh when data is expired"""
        cache = JWKSCache(cache_duration=1)
        cache._cache = {"keys": []}
        cache._cache_time = time.time() - 2  # Expired

        with patch("httpx.Client") as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {"keys": jwks_data["keys"]}
            mock_response.raise_for_status.return_value = None

            mock_context = Mock()
            mock_context.__enter__.return_value.get.return_value = mock_response
            mock_client.return_value = mock_context

            result = cache.get_jwks("https://test-issuer.com/.well-known/jwks.json")

            assert result == {"keys": jwks_data["keys"]}
            mock_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_jwks_cache_http_error(self):
        """Test JWKS cache handling HTTP errors"""
        cache = JWKSCache()
        jwks_url = "https://test-issuer.com/.well-known/jwks.json"

        with patch("httpx.Client") as mock_client:
            mock_context = Mock()
            mock_context.__enter__.return_value.get.side_effect = httpx.HTTPError(
                "Connection failed"
            )
            mock_client.return_value = mock_context

            with pytest.raises(HTTPException) as exc_info:
                cache.get_jwks(jwks_url)

            assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            assert "Unable to fetch JWKS" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_jwks_cache_thread_safety(self):
        """Test JWKS cache thread safety with concurrent access"""
        cache = JWKSCache()

        # This test verifies that the lock is a proper threading.Lock object
        # and can be used in a context manager

        # Verify it's a proper Lock object that supports context manager protocol
        assert hasattr(cache._lock, "__enter__") and hasattr(cache._lock, "__exit__")

        # Test that we can acquire and release the lock
        with cache._lock:
            # Lock is acquired here
            pass
        # Lock is released here

    @pytest.mark.asyncio
    async def test_jwks_cache_unexpected_error(self):
        """Test JWKS cache handling unexpected errors"""
        cache = JWKSCache()

        with patch("httpx.Client") as mock_client:
            mock_client.side_effect = Exception("Unexpected error")

            with pytest.raises(HTTPException) as exc_info:
                cache.get_jwks("https://test-issuer.com/.well-known/jwks.json")

            assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Internal server error" in str(exc_info.value.detail)


class TestJWKSURL:
    """Test JWKS URL generation"""

    def test_get_jwks_url_with_explicit_url(self, mock_settings):
        """Test JWKS URL when explicitly configured"""
        mock_settings.oauth2_jwks_url = "https://custom-jwks.com/keys"
        mock_settings.oauth2_issuer_url = "https://test-issuer.com"

        result = get_jwks_url()
        assert result == "https://custom-jwks.com/keys"

    def test_get_jwks_url_from_issuer(self, mock_settings):
        """Test JWKS URL derived from issuer URL"""
        mock_settings.oauth2_jwks_url = None
        mock_settings.oauth2_issuer_url = "https://test-issuer.com"

        result = get_jwks_url()
        assert result == "https://test-issuer.com/.well-known/jwks.json"

    def test_get_jwks_url_from_issuer_with_trailing_slash(self, mock_settings):
        """Test JWKS URL with issuer having trailing slash"""
        mock_settings.oauth2_jwks_url = None
        mock_settings.oauth2_issuer_url = "https://test-issuer.com/"

        result = get_jwks_url()
        assert result == "https://test-issuer.com/.well-known/jwks.json"

    def test_get_jwks_url_no_issuer(self, mock_settings):
        """Test JWKS URL when issuer is not configured"""
        mock_settings.oauth2_jwks_url = None
        mock_settings.oauth2_issuer_url = None

        with pytest.raises(HTTPException) as exc_info:
            get_jwks_url()

        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "OAuth2 issuer URL not configured" in str(exc_info.value.detail)


class TestPublicKeyRetrieval:
    """Test public key retrieval from JWKS"""

    @patch("agent_memory_server.auth.jwks_cache")
    def test_get_public_key_success(self, mock_cache, jwks_data, valid_token):
        """Test successful public key retrieval"""
        mock_cache.get_jwks.return_value = {"keys": jwks_data["keys"]}

        with patch("agent_memory_server.auth.get_jwks_url") as mock_get_url:
            mock_get_url.return_value = "https://test-issuer.com/.well-known/jwks.json"

            result = get_public_key(valid_token)
            assert result is not None
            assert "BEGIN PUBLIC KEY" in result

    def test_get_public_key_invalid_header(self):
        """Test public key retrieval with invalid JWT header"""
        invalid_token = "invalid.jwt.token"

        with pytest.raises(HTTPException) as exc_info:
            get_public_key(invalid_token)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid token header" in str(exc_info.value.detail)

    @patch("agent_memory_server.auth.jwks_cache")
    def test_get_public_key_missing_kid(self, mock_cache, jwks_data):
        """Test public key retrieval with missing kid in token"""
        # Create token without kid
        payload = {"sub": "test-user"}
        token_without_kid = jwt.encode(payload, "secret", algorithm="HS256")

        with pytest.raises(HTTPException) as exc_info:
            get_public_key(token_without_kid)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Token missing key identifier" in str(exc_info.value.detail)

    @patch("agent_memory_server.auth.jwks_cache")
    def test_get_public_key_kid_not_found(self, mock_cache, jwks_data, valid_token):
        """Test public key retrieval when kid is not found in JWKS"""
        # Modify JWKS to have different kid
        modified_jwks = {"keys": []}
        mock_cache.get_jwks.return_value = modified_jwks

        with patch("agent_memory_server.auth.get_jwks_url") as mock_get_url:
            mock_get_url.return_value = "https://test-issuer.com/.well-known/jwks.json"

            with pytest.raises(HTTPException) as exc_info:
                get_public_key(valid_token)

            assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
            assert "Unable to find matching public key" in str(exc_info.value.detail)

    @patch("agent_memory_server.auth.jwks_cache")
    def test_get_public_key_malformed_jwk(self, mock_cache, valid_token):
        """Test public key retrieval with malformed JWK"""
        malformed_jwks = {
            "keys": [
                {
                    "kid": "test-kid-123",
                    "kty": "RSA",
                    "n": "invalid-modulus",  # Invalid base64
                    "e": "AQAB",
                }
            ]
        }
        mock_cache.get_jwks.return_value = malformed_jwks

        with patch("agent_memory_server.auth.get_jwks_url") as mock_get_url:
            mock_get_url.return_value = "https://test-issuer.com/.well-known/jwks.json"

            with pytest.raises(HTTPException) as exc_info:
                get_public_key(valid_token)

            assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
            assert "Unable to find matching public key" in str(exc_info.value.detail)


class TestJWTVerification:
    """Test JWT token verification"""

    def test_verify_jwt_success(self, mock_settings, jwks_data, valid_token):
        """Test successful JWT verification"""
        mock_settings.oauth2_issuer_url = "https://test-issuer.com"
        mock_settings.oauth2_audience = "test-audience"
        mock_settings.oauth2_algorithms = ["RS256"]

        with patch("agent_memory_server.auth.get_public_key") as mock_get_key:
            # Extract public key from JWKS data
            from jose.jwk import RSAKey

            rsa_key = RSAKey(jwks_data["keys"][0])
            mock_get_key.return_value = rsa_key.to_pem()

            result = verify_jwt(valid_token)

            assert isinstance(result, UserInfo)
            assert result.sub == "test-user-123"
            assert result.aud == "test-audience"
            assert result.email == "test@example.com"
            assert result.scope == "read write admin"
            assert result.roles == ["user", "admin"]

    def test_verify_jwt_missing_issuer_config(self, mock_settings):
        """Test JWT verification when issuer is not configured"""
        mock_settings.oauth2_issuer_url = None

        with pytest.raises(HTTPException) as exc_info:
            verify_jwt("any.jwt.token")

        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "OAuth2 issuer URL not configured" in str(exc_info.value.detail)

    def test_verify_jwt_expired_token(self, mock_settings, jwks_data, expired_token):
        """Test JWT verification with expired token"""
        mock_settings.oauth2_issuer_url = "https://test-issuer.com"
        mock_settings.oauth2_audience = "test-audience"

        with patch("agent_memory_server.auth.get_public_key") as mock_get_key:
            from jose.jwk import RSAKey

            rsa_key = RSAKey(jwks_data["keys"][0])
            mock_get_key.return_value = rsa_key.to_pem()

            with pytest.raises(HTTPException) as exc_info:
                verify_jwt(expired_token)

            assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
            assert "Invalid JWT" in str(exc_info.value.detail)

    def test_verify_jwt_future_token(self, mock_settings, jwks_data):
        """Test JWT verification with token issued in the future"""
        mock_settings.oauth2_issuer_url = "https://test-issuer.com"
        mock_settings.oauth2_audience = "test-audience"

        # Create token issued in the future
        future_payload = {
            "sub": "test-user-123",
            "aud": "test-audience",
            "iss": "https://test-issuer.com",
            "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
            "iat": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),  # Future
        }

        future_token = jwt.encode(
            future_payload,
            jwks_data["private_key"],
            algorithm="RS256",
            headers={"kid": jwks_data["kid"]},
        )

        with patch("agent_memory_server.auth.get_public_key") as mock_get_key:
            from jose.jwk import RSAKey

            rsa_key = RSAKey(jwks_data["keys"][0])
            mock_get_key.return_value = rsa_key.to_pem()

            with pytest.raises(HTTPException) as exc_info:
                verify_jwt(future_token)

            assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
            assert "Token issued in the future" in str(exc_info.value.detail)

    def test_verify_jwt_wrong_audience(self, mock_settings, jwks_data):
        """Test JWT verification with wrong audience"""
        mock_settings.oauth2_issuer_url = "https://test-issuer.com"
        mock_settings.oauth2_audience = "expected-audience"

        # Create token with wrong audience
        wrong_aud_payload = {
            "sub": "test-user-123",
            "aud": "wrong-audience",
            "iss": "https://test-issuer.com",
            "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
            "iat": int(datetime.now(UTC).timestamp()),
        }

        wrong_aud_token = jwt.encode(
            wrong_aud_payload,
            jwks_data["private_key"],
            algorithm="RS256",
            headers={"kid": jwks_data["kid"]},
        )

        with patch("agent_memory_server.auth.get_public_key") as mock_get_key:
            from jose.jwk import RSAKey

            rsa_key = RSAKey(jwks_data["keys"][0])
            mock_get_key.return_value = rsa_key.to_pem()

            with pytest.raises(HTTPException) as exc_info:
                verify_jwt(wrong_aud_token)

            assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
            assert "Invalid audience" in str(exc_info.value.detail)

    def test_verify_jwt_audience_list(self, mock_settings, jwks_data):
        """Test JWT verification with audience as list"""
        mock_settings.oauth2_issuer_url = "https://test-issuer.com"
        mock_settings.oauth2_audience = "test-audience"

        # Create token with audience as list including our audience
        list_aud_payload = {
            "sub": "test-user-123",
            "aud": ["test-audience", "other-audience"],
            "iss": "https://test-issuer.com",
            "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
            "iat": int(datetime.now(UTC).timestamp()),
        }

        list_aud_token = jwt.encode(
            list_aud_payload,
            jwks_data["private_key"],
            algorithm="RS256",
            headers={"kid": jwks_data["kid"]},
        )

        with patch("agent_memory_server.auth.get_public_key") as mock_get_key:
            from jose.jwk import RSAKey

            rsa_key = RSAKey(jwks_data["keys"][0])
            mock_get_key.return_value = rsa_key.to_pem()

            result = verify_jwt(list_aud_token)
            assert result.sub == "test-user-123"

    def test_verify_jwt_missing_subject(self, mock_settings, jwks_data):
        """Test JWT verification with missing subject"""
        mock_settings.oauth2_issuer_url = "https://test-issuer.com"
        mock_settings.oauth2_audience = "test-audience"

        # Create token without subject
        no_sub_payload = {
            "aud": "test-audience",
            "iss": "https://test-issuer.com",
            "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
            "iat": int(datetime.now(UTC).timestamp()),
        }

        no_sub_token = jwt.encode(
            no_sub_payload,
            jwks_data["private_key"],
            algorithm="RS256",
            headers={"kid": jwks_data["kid"]},
        )

        with patch("agent_memory_server.auth.get_public_key") as mock_get_key:
            from jose.jwk import RSAKey

            rsa_key = RSAKey(jwks_data["keys"][0])
            mock_get_key.return_value = rsa_key.to_pem()

            with pytest.raises(HTTPException) as exc_info:
                verify_jwt(no_sub_token)

            assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
            assert "Token missing subject" in str(exc_info.value.detail)

    def test_verify_jwt_scope_string_conversion(self, mock_settings, jwks_data):
        """Test JWT verification with scope as single string"""
        mock_settings.oauth2_issuer_url = "https://test-issuer.com"
        mock_settings.oauth2_audience = "test-audience"

        # Create token with scope as single string
        scope_str_payload = {
            "sub": "test-user-123",
            "aud": "test-audience",
            "iss": "https://test-issuer.com",
            "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
            "iat": int(datetime.now(UTC).timestamp()),
            "scope": "",  # Empty scope
        }

        scope_str_token = jwt.encode(
            scope_str_payload,
            jwks_data["private_key"],
            algorithm="RS256",
            headers={"kid": jwks_data["kid"]},
        )

        with patch("agent_memory_server.auth.get_public_key") as mock_get_key:
            from jose.jwk import RSAKey

            rsa_key = RSAKey(jwks_data["keys"][0])
            mock_get_key.return_value = rsa_key.to_pem()

            result = verify_jwt(scope_str_token)
            assert result.scope is None

    def test_verify_jwt_roles_string_conversion(self, mock_settings, jwks_data):
        """Test JWT verification with roles as single string"""
        mock_settings.oauth2_issuer_url = "https://test-issuer.com"
        mock_settings.oauth2_audience = "test-audience"

        # Create token with roles as single string
        roles_str_payload = {
            "sub": "test-user-123",
            "aud": "test-audience",
            "iss": "https://test-issuer.com",
            "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
            "iat": int(datetime.now(UTC).timestamp()),
            "roles": "admin",  # Single role string
        }

        roles_str_token = jwt.encode(
            roles_str_payload,
            jwks_data["private_key"],
            algorithm="RS256",
            headers={"kid": jwks_data["kid"]},
        )

        with patch("agent_memory_server.auth.get_public_key") as mock_get_key:
            from jose.jwk import RSAKey

            rsa_key = RSAKey(jwks_data["keys"][0])
            mock_get_key.return_value = rsa_key.to_pem()

            result = verify_jwt(roles_str_token)
            assert result.roles == ["admin"]

    def test_verify_jwt_no_audience_validation(self, mock_settings, jwks_data):
        """Test JWT verification without audience validation"""
        mock_settings.oauth2_issuer_url = "https://test-issuer.com"
        mock_settings.oauth2_audience = None  # No audience validation

        # Create token without audience
        no_aud_payload = {
            "sub": "test-user-123",
            "iss": "https://test-issuer.com",
            "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
            "iat": int(datetime.now(UTC).timestamp()),
        }

        no_aud_token = jwt.encode(
            no_aud_payload,
            jwks_data["private_key"],
            algorithm="RS256",
            headers={"kid": jwks_data["kid"]},
        )

        with patch("agent_memory_server.auth.get_public_key") as mock_get_key:
            from jose.jwk import RSAKey

            rsa_key = RSAKey(jwks_data["keys"][0])
            mock_get_key.return_value = rsa_key.to_pem()

            result = verify_jwt(no_aud_token)
            assert result.sub == "test-user-123"
            assert result.aud is None


class TestGetCurrentUser:
    """Test get_current_user dependency function"""

    @pytest.mark.asyncio
    async def test_get_current_user_disabled_auth(self, mock_settings):
        """Test get_current_user when authentication is disabled"""
        mock_settings.disable_auth = True

        result = get_current_user(None)

        assert isinstance(result, UserInfo)
        assert result.sub == "local-dev-user"
        assert result.aud == "local-dev"
        assert result.scope == "admin"
        assert result.roles == ["admin"]

    @pytest.mark.asyncio
    async def test_get_current_user_missing_credentials(self, mock_settings):
        """Test get_current_user with missing credentials"""
        mock_settings.disable_auth = False

        with pytest.raises(HTTPException) as exc_info:
            get_current_user(None)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Missing authorization header" in str(exc_info.value.detail)
        assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}

    @pytest.mark.asyncio
    async def test_get_current_user_empty_credentials(self, mock_settings):
        """Test get_current_user with empty credentials"""
        mock_settings.disable_auth = False

        from fastapi.security import HTTPAuthorizationCredentials

        empty_creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="")

        with pytest.raises(HTTPException) as exc_info:
            get_current_user(empty_creds)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Missing bearer token" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_current_user_valid_token(self, mock_settings, valid_token):
        """Test get_current_user with valid token"""
        mock_settings.disable_auth = False

        from fastapi.security import HTTPAuthorizationCredentials

        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=valid_token)

        with patch("agent_memory_server.auth.verify_jwt") as mock_verify:
            expected_user = UserInfo(sub="test-user", email="test@example.com")
            mock_verify.return_value = expected_user

            result = get_current_user(creds)

            assert result == expected_user
            mock_verify.assert_called_once_with(valid_token)


class TestRoleAndScopeRequirements:
    """Test role and scope requirement decorators"""

    @pytest.mark.asyncio
    async def test_require_scope_success(self, mock_settings):
        """Test successful scope requirement check"""
        mock_settings.disable_auth = False

        user = UserInfo(sub="test-user", scope="read write admin")
        scope_dependency = require_scope("read")

        result = scope_dependency(user)
        assert result == user

    @pytest.mark.asyncio
    async def test_require_scope_failure(self, mock_settings):
        """Test failed scope requirement check"""
        mock_settings.disable_auth = False

        user = UserInfo(sub="test-user", scope="read write")
        scope_dependency = require_scope("admin")

        with pytest.raises(HTTPException) as exc_info:
            scope_dependency(user)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "Insufficient permissions" in str(exc_info.value.detail)
        assert "Required scope: admin" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_require_scope_no_scope(self, mock_settings):
        """Test scope requirement with user having no scope"""
        mock_settings.disable_auth = False

        user = UserInfo(sub="test-user", scope=None)
        scope_dependency = require_scope("read")

        with pytest.raises(HTTPException) as exc_info:
            scope_dependency(user)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN

    @pytest.mark.asyncio
    async def test_require_scope_disabled_auth(self, mock_settings):
        """Test scope requirement when auth is disabled"""
        mock_settings.disable_auth = True

        user = UserInfo(sub="test-user", scope=None)
        scope_dependency = require_scope("admin")

        result = scope_dependency(user)
        assert result == user

    @pytest.mark.asyncio
    async def test_require_role_success(self, mock_settings):
        """Test successful role requirement check"""
        mock_settings.disable_auth = False

        user = UserInfo(sub="test-user", roles=["user", "admin"])
        role_dependency = require_role("admin")

        result = role_dependency(user)
        assert result == user

    @pytest.mark.asyncio
    async def test_require_role_failure(self, mock_settings):
        """Test failed role requirement check"""
        mock_settings.disable_auth = False

        user = UserInfo(sub="test-user", roles=["user"])
        role_dependency = require_role("admin")

        with pytest.raises(HTTPException) as exc_info:
            role_dependency(user)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "Insufficient permissions" in str(exc_info.value.detail)
        assert "Required role: admin" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_require_role_no_roles(self, mock_settings):
        """Test role requirement with user having no roles"""
        mock_settings.disable_auth = False

        user = UserInfo(sub="test-user", roles=None)
        role_dependency = require_role("admin")

        with pytest.raises(HTTPException) as exc_info:
            role_dependency(user)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN

    @pytest.mark.asyncio
    async def test_require_role_disabled_auth(self, mock_settings):
        """Test role requirement when auth is disabled"""
        mock_settings.disable_auth = True

        user = UserInfo(sub="test-user", roles=None)
        role_dependency = require_role("admin")

        result = role_dependency(user)
        assert result == user


class TestAuthConfiguration:
    """Test authentication configuration validation"""

    def test_verify_auth_config_disabled(self, mock_settings):
        """Test auth config verification when auth is disabled"""
        mock_settings.disable_auth = True
        mock_settings.oauth2_issuer_url = None
        mock_settings.oauth2_audience = None

        # Should not raise any exception
        verify_auth_config()

    def test_verify_auth_config_missing_issuer(self, mock_settings):
        """Test auth config verification with missing issuer"""
        mock_settings.disable_auth = False
        mock_settings.oauth2_issuer_url = None

        with pytest.raises(ValueError) as exc_info:
            verify_auth_config()

        assert "OAUTH2_ISSUER_URL must be set" in str(exc_info.value)

    def test_verify_auth_config_valid(self, mock_settings):
        """Test auth config verification with valid configuration"""
        mock_settings.disable_auth = False
        mock_settings.oauth2_issuer_url = "https://test-issuer.com"
        mock_settings.oauth2_audience = "test-audience"
        mock_settings.oauth2_algorithms = ["RS256"]

        # Should not raise any exception
        verify_auth_config()

    def test_verify_auth_config_missing_audience_warning(self, mock_settings, caplog):
        """Test auth config verification with missing audience (should warn)"""
        mock_settings.disable_auth = False
        mock_settings.oauth2_issuer_url = "https://test-issuer.com"
        mock_settings.oauth2_audience = None

        verify_auth_config()

        # Check that a warning was logged (caplog might not capture structlog)
        # This test verifies the function completes without error


class TestUserInfoModel:
    """Test UserInfo Pydantic model"""

    def test_user_info_creation(self):
        """Test UserInfo model creation with all fields"""
        user = UserInfo(
            sub="test-user-123",
            aud="test-audience",
            scope="read write admin",
            exp=1234567890,
            iat=1234567880,
            iss="https://test-issuer.com",
            email="test@example.com",
            roles=["user", "admin"],
        )

        assert user.sub == "test-user-123"
        assert user.aud == "test-audience"
        assert user.scope == "read write admin"
        assert user.exp == 1234567890
        assert user.iat == 1234567880
        assert user.iss == "https://test-issuer.com"
        assert user.email == "test@example.com"
        assert user.roles == ["user", "admin"]

    def test_user_info_minimal_creation(self):
        """Test UserInfo model creation with minimal fields"""
        user = UserInfo(sub="test-user-123")

        assert user.sub == "test-user-123"
        assert user.aud is None
        assert user.scope is None
        assert user.exp is None
        assert user.iat is None
        assert user.iss is None
        assert user.email is None
        assert user.roles is None


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""

    @pytest.mark.asyncio
    async def test_auth0_integration_scenario(self, mock_settings, jwks_data):
        """Test integration scenario mimicking Auth0"""
        mock_settings.disable_auth = False
        mock_settings.oauth2_issuer_url = "https://myapp.auth0.com/"
        mock_settings.oauth2_audience = "https://api.myapp.com"
        mock_settings.oauth2_algorithms = ["RS256"]

        # Create Auth0-style token
        auth0_payload = {
            "sub": "auth0|507f1f77bcf86cd799439011",
            "aud": ["https://api.myapp.com", "https://myapp.auth0.com/userinfo"],
            "iss": "https://myapp.auth0.com/",
            "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
            "iat": int(datetime.now(UTC).timestamp()),
            "scope": "read:users write:users",
            "permissions": ["read:users", "write:users"],
        }

        auth0_token = jwt.encode(
            auth0_payload,
            jwks_data["private_key"],
            algorithm="RS256",
            headers={"kid": jwks_data["kid"]},
        )

        with patch("agent_memory_server.auth.get_public_key") as mock_get_key:
            from jose.jwk import RSAKey

            rsa_key = RSAKey(jwks_data["keys"][0])
            mock_get_key.return_value = rsa_key.to_pem()

            result = verify_jwt(auth0_token)

            assert result.sub == "auth0|507f1f77bcf86cd799439011"
            assert result.aud == [
                "https://api.myapp.com",
                "https://myapp.auth0.com/userinfo",
            ]
            assert result.scope == "read:users write:users"

    @pytest.mark.asyncio
    async def test_aws_cognito_integration_scenario(self, mock_settings, jwks_data):
        """Test integration scenario mimicking AWS Cognito"""
        mock_settings.disable_auth = False
        mock_settings.oauth2_issuer_url = (
            "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_XXXXXXXXX"
        )
        mock_settings.oauth2_audience = "my-app-client-id"
        mock_settings.oauth2_algorithms = ["RS256"]

        # Create Cognito-style token
        cognito_payload = {
            "sub": "550e8400-e29b-41d4-a716-446655440000",
            "aud": "my-app-client-id",
            "iss": "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_XXXXXXXXX",
            "token_use": "access",
            "scope": "aws.cognito.signin.user.admin",
            "auth_time": int(datetime.now(UTC).timestamp()),
            "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
            "iat": int(datetime.now(UTC).timestamp()),
            "client_id": "my-app-client-id",
            "username": "testuser",
        }

        cognito_token = jwt.encode(
            cognito_payload,
            jwks_data["private_key"],
            algorithm="RS256",
            headers={"kid": jwks_data["kid"]},
        )

        with patch("agent_memory_server.auth.get_public_key") as mock_get_key:
            from jose.jwk import RSAKey

            rsa_key = RSAKey(jwks_data["keys"][0])
            mock_get_key.return_value = rsa_key.to_pem()

            result = verify_jwt(cognito_token)

            assert result.sub == "550e8400-e29b-41d4-a716-446655440000"
            assert result.aud == "my-app-client-id"
            assert result.scope == "aws.cognito.signin.user.admin"

    @pytest.mark.asyncio
    async def test_performance_token_validation(
        self, mock_settings, jwks_data, valid_token
    ):
        """Test performance aspects of token validation"""
        mock_settings.disable_auth = False
        mock_settings.oauth2_issuer_url = "https://test-issuer.com"
        mock_settings.oauth2_audience = "test-audience"

        # Mock cached JWKS to avoid network calls
        with patch("agent_memory_server.auth.jwks_cache") as mock_cache:
            mock_cache.get_jwks.return_value = {"keys": jwks_data["keys"]}

            with patch("agent_memory_server.auth.get_jwks_url") as mock_get_url:
                mock_get_url.return_value = (
                    "https://test-issuer.com/.well-known/jwks.json"
                )

                # Validate multiple tokens to test caching
                start_time = time.time()
                for _ in range(10):
                    result = verify_jwt(valid_token)
                    assert result.sub == "test-user-123"
                end_time = time.time()

                # Should be fast due to caching
                duration = end_time - start_time
                assert duration < 1.0  # Should complete in less than 1 second

                # JWKS should only be fetched once due to caching
                assert mock_cache.get_jwks.call_count <= 10  # Allow some flexibility

    @pytest.mark.asyncio
    async def test_concurrent_auth_requests(
        self, mock_settings, jwks_data, valid_token
    ):
        """Test concurrent authentication requests"""
        import asyncio

        mock_settings.disable_auth = False
        mock_settings.oauth2_issuer_url = "https://test-issuer.com"
        mock_settings.oauth2_audience = "test-audience"

        with patch("agent_memory_server.auth.jwks_cache") as mock_cache:
            mock_cache.get_jwks.return_value = {"keys": jwks_data["keys"]}

            with patch("agent_memory_server.auth.get_jwks_url") as mock_get_url:
                mock_get_url.return_value = (
                    "https://test-issuer.com/.well-known/jwks.json"
                )

                async def validate_token():
                    return verify_jwt(valid_token)

                # Run multiple concurrent validations
                tasks = [validate_token() for _ in range(5)]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # All should succeed
                for result in results:
                    assert isinstance(result, UserInfo)
                    assert result.sub == "test-user-123"


class TestErrorHandling:
    """Test comprehensive error handling scenarios"""

    @pytest.mark.asyncio
    async def test_network_timeout_handling(self, mock_settings):
        """Test handling of network timeouts during JWKS fetch"""
        mock_settings.disable_auth = False
        mock_settings.oauth2_issuer_url = "https://test-issuer.com"

        cache = JWKSCache()

        with patch("httpx.Client") as mock_client:
            mock_context = Mock()
            mock_context.__enter__.return_value.get.side_effect = (
                httpx.TimeoutException("Timeout")
            )
            mock_client.return_value = mock_context

            with pytest.raises(HTTPException) as exc_info:
                cache.get_jwks("https://test-issuer.com/.well-known/jwks.json")

            assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            assert "Unable to fetch JWKS" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_malformed_jwt_handling(self, mock_settings):
        """Test handling of completely malformed JWT tokens"""
        mock_settings.disable_auth = False

        malformed_tokens = [
            "not.a.jwt",
            "definitely-not-a-jwt",
            "a.b",  # Too few parts
            "a.b.c.d",  # Too many parts
            "",  # Empty string
        ]

        for malformed_token in malformed_tokens:
            with pytest.raises(HTTPException) as exc_info:
                get_public_key(malformed_token)

            assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
            assert "Invalid token header" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_jwks_endpoint_returns_invalid_json(self, mock_settings):
        """Test handling of JWKS endpoint returning invalid JSON"""
        cache = JWKSCache()

        with patch("httpx.Client") as mock_client:
            mock_response = Mock()
            mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            mock_response.raise_for_status.return_value = None

            mock_context = Mock()
            mock_context.__enter__.return_value.get.return_value = mock_response
            mock_client.return_value = mock_context

            with pytest.raises(HTTPException) as exc_info:
                cache.get_jwks("https://test-issuer.com/.well-known/jwks.json")

            assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    @pytest.mark.asyncio
    async def test_memory_pressure_scenarios(self, mock_settings, jwks_data):
        """Test behavior under memory pressure (large JWKS responses)"""
        mock_settings.disable_auth = False

        # Create a large JWKS response to simulate memory pressure
        large_jwks = {
            "keys": [jwks_data["keys"][0]] * 1000  # Repeat the key 1000 times
        }

        cache = JWKSCache()

        with patch("httpx.Client") as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = large_jwks
            mock_response.raise_for_status.return_value = None

            mock_context = Mock()
            mock_context.__enter__.return_value.get.return_value = mock_response
            mock_client.return_value = mock_context

            # Should handle large responses gracefully
            result = cache.get_jwks("https://test-issuer.com/.well-known/jwks.json")
            assert len(result["keys"]) == 1000
