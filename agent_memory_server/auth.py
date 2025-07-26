import secrets
import threading
import time
from datetime import UTC, datetime
from typing import Any

import bcrypt
import httpx
import structlog
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwk, jwt
from pydantic import BaseModel

from agent_memory_server.config import settings
from agent_memory_server.utils.keys import Keys
from agent_memory_server.utils.redis import get_redis_conn


logger = structlog.get_logger()


class UserInfo(BaseModel):
    sub: str
    aud: str | list[str] | None = None
    scope: str | None = None
    exp: int | None = None
    iat: int | None = None
    iss: str | None = None
    email: str | None = None
    roles: list[str] | None = None


class TokenInfo(BaseModel):
    """Token information stored in Redis."""

    description: str
    created_at: datetime
    expires_at: datetime | None = None
    token_hash: str


class JWKSCache:
    def __init__(self, cache_duration: int = 3600):
        self._cache: dict[str, Any] = {}
        self._cache_time: float | None = None
        self._cache_duration = cache_duration
        self._lock = threading.Lock()

    def get_jwks(self, jwks_url: str) -> dict[str, Any]:
        current_time = time.time()

        if (
            self._cache_time is None
            or current_time - self._cache_time > self._cache_duration
            or not self._cache
        ):
            with self._lock:
                # Double-check pattern: another thread might have updated cache while waiting
                if (
                    self._cache_time is not None
                    and current_time - self._cache_time <= self._cache_duration
                    and self._cache
                ):
                    return self._cache

                try:
                    logger.info("Fetching JWKS keys", jwks_url=jwks_url)

                    with httpx.Client(timeout=10.0) as client:
                        response = client.get(jwks_url)
                        response.raise_for_status()

                    jwks_data = response.json()
                    self._cache = jwks_data
                    self._cache_time = current_time

                    logger.info(
                        "Successfully cached JWKS keys",
                        key_count=len(jwks_data.get("keys", [])),
                    )

                except httpx.HTTPError as e:
                    logger.error(
                        "Failed to fetch JWKS", error=str(e), jwks_url=jwks_url
                    )
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"Unable to fetch JWKS from {jwks_url}: {str(e)}",
                    ) from e
                except Exception as e:
                    logger.error("Unexpected error fetching JWKS", error=str(e))
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Internal server error while fetching JWKS",
                    ) from e

        return self._cache


jwks_cache = JWKSCache()
oauth2_scheme = HTTPBearer(auto_error=False)


def get_jwks_url() -> str:
    if settings.oauth2_jwks_url:
        return settings.oauth2_jwks_url

    if not settings.oauth2_issuer_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OAuth2 issuer URL not configured",
        )

    issuer_url = settings.oauth2_issuer_url.rstrip("/")
    return f"{issuer_url}/.well-known/jwks.json"


def get_public_key(token: str) -> str:
    try:
        unverified_header = jwt.get_unverified_header(token)
    except JWTError as e:
        logger.warning("Invalid JWT header", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token header"
        ) from e

    kid = unverified_header.get("kid")
    if not kid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing key identifier (kid)",
        )

    jwks_url = get_jwks_url()
    jwks_data = jwks_cache.get_jwks(jwks_url)

    keys = jwks_data.get("keys", [])
    public_key = None

    for key in keys:
        if key.get("kid") == kid:
            try:
                public_key_bytes = jwk.construct(key).to_pem()
                public_key = public_key_bytes.decode("utf-8")
                break
            except Exception as e:
                logger.error("Failed to construct public key", kid=kid, error=str(e))
                continue

    if not public_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Unable to find matching public key for kid: {kid}",
        )

    return public_key


def verify_jwt(token: str) -> UserInfo:
    if not settings.oauth2_issuer_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OAuth2 issuer URL not configured",
        )

    try:
        public_key = get_public_key(token)

        decode_options = {
            "verify_signature": True,
            "verify_exp": True,
            "verify_nbf": True,
            "verify_iat": True,
            "verify_aud": bool(settings.oauth2_audience),
            "require_exp": True,
            "require_iat": True,
        }

        payload = jwt.decode(
            token,
            public_key,
            algorithms=settings.oauth2_algorithms,
            audience=settings.oauth2_audience,
            issuer=settings.oauth2_issuer_url,
            options=decode_options,
        )

        current_time = int(datetime.now(UTC).timestamp())

        exp = payload.get("exp")
        if exp and exp < current_time:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired"
            )

        iat = payload.get("iat")
        if iat and iat > current_time + 300:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token issued in the future",
            )

        if settings.oauth2_audience:
            aud = payload.get("aud")
            if isinstance(aud, list):
                if settings.oauth2_audience not in aud:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail=f"Invalid audience. Expected: {settings.oauth2_audience}",
                    )
            elif aud != settings.oauth2_audience:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid audience. Expected: {settings.oauth2_audience}",
                )

        sub = payload.get("sub")
        if not sub:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing subject (sub) claim",
            )

        scope = payload.get("scope", "")
        if isinstance(scope, str):
            scope = scope.split() if scope else []

        roles = payload.get("roles", [])
        if isinstance(roles, str):
            roles = [roles]

        return UserInfo(
            sub=sub,
            aud=payload.get("aud"),
            scope=" ".join(scope) if scope else None,
            exp=payload.get("exp"),
            iat=payload.get("iat"),
            iss=payload.get("iss"),
            email=payload.get("email"),
            roles=roles,
        )

    except HTTPException:
        raise
    except JWTError as e:
        logger.warning(
            "JWT validation failed", error=str(e), token_prefix=token[:20] + "..."
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid JWT: {str(e)}"
        ) from e
    except Exception as e:
        logger.error("Unexpected error during JWT validation", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during authentication",
        ) from e


def generate_token() -> str:
    """Generate a secure random token."""
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    """Hash a token using bcrypt."""
    return bcrypt.hashpw(token.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_token_hash(token: str, token_hash: str) -> bool:
    """Verify a token against its hash."""
    try:
        return bcrypt.checkpw(token.encode("utf-8"), token_hash.encode("utf-8"))
    except Exception as e:
        logger.warning("Token hash verification failed", error=str(e))
        return False


async def verify_token(token: str) -> UserInfo:
    """Verify a token and return user info."""
    try:
        redis = await get_redis_conn()

        # Get all auth tokens and check each one
        # This is not the most efficient approach, but it works for now
        # In a production system, you might want to store a mapping of token prefixes
        pattern = Keys.auth_token_key("*")
        token_keys = []

        async for key in redis.scan_iter(pattern):
            token_keys.append(key)

        for key in token_keys:
            token_data = await redis.get(key)
            if not token_data:
                continue

            try:
                token_info = TokenInfo.model_validate_json(token_data)

                # Check if token matches
                if verify_token_hash(token, token_info.token_hash):
                    # Check if token is expired
                    if (
                        token_info.expires_at
                        and datetime.now(UTC) > token_info.expires_at
                    ):
                        logger.warning("Token has expired")
                        raise HTTPException(
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Token has expired",
                        )

                    # Return user info for valid token
                    return UserInfo(
                        sub="token-user",
                        aud="token-auth",
                        scope="admin",
                        roles=["admin"],
                        exp=int(token_info.expires_at.timestamp())
                        if token_info.expires_at
                        else None,
                        iat=int(token_info.created_at.timestamp()),
                    )

            except HTTPException:
                # Re-raise HTTP exceptions (like token expired)
                raise
            except Exception as e:
                logger.warning("Error processing token", error=str(e))
                continue

        # If no token matched, authentication failed
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error during token verification", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during authentication",
        ) from e


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(oauth2_scheme),
) -> UserInfo:
    if settings.disable_auth or settings.auth_mode == "disabled":
        logger.debug("Authentication disabled, returning default user")
        return UserInfo(
            sub="local-dev-user", aud="local-dev", scope="admin", roles=["admin"]
        )

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Determine authentication mode
    if settings.auth_mode == "token" or settings.token_auth_enabled:
        return await verify_token(credentials.credentials)
    if settings.auth_mode == "oauth2":
        return verify_jwt(credentials.credentials)
    # Default to OAuth2 for backward compatibility
    return verify_jwt(credentials.credentials)


def require_scope(required_scope: str):
    async def scope_dependency(user: UserInfo = Depends(get_current_user)) -> UserInfo:
        if settings.disable_auth:
            return user

        user_scopes = user.scope.split() if user.scope else []
        if required_scope not in user_scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required scope: {required_scope}",
            )
        return user

    return scope_dependency


def require_role(required_role: str):
    async def role_dependency(user: UserInfo = Depends(get_current_user)) -> UserInfo:
        if settings.disable_auth:
            return user

        user_roles = user.roles or []
        if required_role not in user_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role}",
            )
        return user

    return role_dependency


def verify_auth_config():
    if settings.disable_auth or settings.auth_mode == "disabled":
        logger.warning("Authentication is DISABLED - suitable for development only")
        return

    if settings.auth_mode == "token" or settings.token_auth_enabled:
        logger.info("Token authentication configured")
        return

    if settings.auth_mode == "oauth2":
        if not settings.oauth2_issuer_url:
            raise ValueError(
                "OAUTH2_ISSUER_URL must be set when OAuth2 authentication is enabled"
            )

        if not settings.oauth2_audience:
            logger.warning(
                "OAUTH2_AUDIENCE not set - audience validation will be skipped"
            )

        logger.info(
            "OAuth2 authentication configured",
            issuer=settings.oauth2_issuer_url,
            audience=settings.oauth2_audience or "not-set",
            algorithms=settings.oauth2_algorithms,
        )
        return

    # Default to OAuth2 for backward compatibility
    if not settings.oauth2_issuer_url:
        raise ValueError("OAUTH2_ISSUER_URL must be set when authentication is enabled")

    if not settings.oauth2_audience:
        logger.warning("OAUTH2_AUDIENCE not set - audience validation will be skipped")

    logger.info(
        "OAuth2 authentication configured (default)",
        issuer=settings.oauth2_issuer_url,
        audience=settings.oauth2_audience or "not-set",
        algorithms=settings.oauth2_algorithms,
    )
