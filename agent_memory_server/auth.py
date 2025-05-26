import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx
import structlog
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt, jwk, JWTError
from pydantic import BaseModel

from agent_memory_server.config import settings


logger = structlog.get_logger()


class UserInfo(BaseModel):
    sub: str
    aud: Optional[str] = None
    scope: Optional[str] = None
    exp: Optional[int] = None
    iat: Optional[int] = None
    iss: Optional[str] = None
    email: Optional[str] = None
    roles: Optional[list[str]] = None


class JWKSCache:
    def __init__(self, cache_duration: int = 3600):
        self._cache: Dict[str, Any] = {}
        self._cache_time: Optional[float] = None
        self._cache_duration = cache_duration
        self._lock = False

    def get_jwks(self, jwks_url: str) -> Dict[str, Any]:
        current_time = time.time()
        
        if (self._cache_time is None or 
            current_time - self._cache_time > self._cache_duration or
            not self._cache):
            
            if self._lock:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="JWKS refresh in progress, try again later"
                )
            
            try:
                self._lock = True
                logger.info("Fetching JWKS keys", jwks_url=jwks_url)
                
                with httpx.Client(timeout=10.0) as client:
                    response = client.get(jwks_url)
                    response.raise_for_status()
                    
                jwks_data = response.json()
                self._cache = jwks_data
                self._cache_time = current_time
                
                logger.info("Successfully cached JWKS keys", 
                           key_count=len(jwks_data.get("keys", [])))
                           
            except httpx.HTTPError as e:
                logger.error("Failed to fetch JWKS", error=str(e), jwks_url=jwks_url)
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Unable to fetch JWKS from {jwks_url}: {str(e)}"
                )
            except Exception as e:
                logger.error("Unexpected error fetching JWKS", error=str(e))
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while fetching JWKS"
                )
            finally:
                self._lock = False
                
        return self._cache


jwks_cache = JWKSCache()
oauth2_scheme = HTTPBearer(auto_error=False)


def get_jwks_url() -> str:
    if settings.oauth2_jwks_url:
        return settings.oauth2_jwks_url
    
    if not settings.oauth2_issuer_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OAuth2 issuer URL not configured"
        )
    
    issuer_url = settings.oauth2_issuer_url.rstrip("/")
    return f"{issuer_url}/.well-known/jwks.json"


def get_public_key(token: str) -> str:
    try:
        unverified_header = jwt.get_unverified_header(token)
    except JWTError as e:
        logger.warning("Invalid JWT header", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token header"
        )
    
    kid = unverified_header.get("kid")
    if not kid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing key identifier (kid)"
        )
    
    jwks_url = get_jwks_url()
    jwks_data = jwks_cache.get_jwks(jwks_url)
    
    keys = jwks_data.get("keys", [])
    public_key = None
    
    for key in keys:
        if key.get("kid") == kid:
            try:
                public_key = jwk.construct(key).to_pem()
                break
            except Exception as e:
                logger.error("Failed to construct public key", kid=kid, error=str(e))
                continue
    
    if not public_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Unable to find matching public key for kid: {kid}"
        )
    
    return public_key


def verify_jwt(token: str) -> UserInfo:
    if not settings.oauth2_issuer_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OAuth2 issuer URL not configured"
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
            options=decode_options
        )
        
        current_time = int(datetime.now(timezone.utc).timestamp())
        
        exp = payload.get("exp")
        if exp and exp < current_time:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        
        iat = payload.get("iat")
        if iat and iat > current_time + 300:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token issued in the future"
            )
        
        if settings.oauth2_audience:
            aud = payload.get("aud")
            if isinstance(aud, list):
                if settings.oauth2_audience not in aud:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail=f"Invalid audience. Expected: {settings.oauth2_audience}"
                    )
            elif aud != settings.oauth2_audience:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid audience. Expected: {settings.oauth2_audience}"
                )
        
        sub = payload.get("sub")
        if not sub:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing subject (sub) claim"
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
            roles=roles
        )
        
    except HTTPException:
        raise
    except JWTError as e:
        logger.warning("JWT validation failed", error=str(e), token_prefix=token[:20] + "...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid JWT: {str(e)}"
        )
    except Exception as e:
        logger.error("Unexpected error during JWT validation", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during authentication"
        )


def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(oauth2_scheme)) -> UserInfo:
    if settings.disable_auth:
        logger.debug("Authentication disabled, returning default user")
        return UserInfo(
            sub="local-dev-user",
            aud="local-dev",
            scope="admin",
            roles=["admin"]
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
    
    return verify_jwt(credentials.credentials)


def require_scope(required_scope: str):
    def scope_dependency(user: UserInfo = Depends(get_current_user)) -> UserInfo:
        if settings.disable_auth:
            return user
        
        user_scopes = user.scope.split() if user.scope else []
        if required_scope not in user_scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required scope: {required_scope}"
            )
        return user
    
    return scope_dependency


def require_role(required_role: str):
    def role_dependency(user: UserInfo = Depends(get_current_user)) -> UserInfo:
        if settings.disable_auth:
            return user
        
        user_roles = user.roles or []
        if required_role not in user_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role}"
            )
        return user
    
    return role_dependency


def verify_auth_config():
    if settings.disable_auth:
        logger.warning("Authentication is DISABLED - suitable for development only")
        return
    
    if not settings.oauth2_issuer_url:
        raise ValueError("OAUTH2_ISSUER_URL must be set when authentication is enabled")
    
    if not settings.oauth2_audience:
        logger.warning("OAUTH2_AUDIENCE not set - audience validation will be skipped")
    
    logger.info("OAuth2 authentication configured", 
               issuer=settings.oauth2_issuer_url,
               audience=settings.oauth2_audience or "not-set",
               algorithms=settings.oauth2_algorithms)