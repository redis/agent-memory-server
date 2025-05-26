import os
from typing import Dict, Any
from urllib.parse import urljoin

import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from jose.exceptions import JWKError

from agent_memory_server.config import settings
from agent_memory_server.logging import get_logger


logger = get_logger(__name__)

oauth2_scheme = HTTPBearer(auto_error=False)

# Cache for JWKS keys
_jwks_cache: Dict[str, Any] = {}


async def get_jwks_keys() -> Dict[str, Any]:
    """
    Fetch and cache JWKS keys from the OAuth2 issuer.
    
    Returns:
        Dict containing JWKS keys
        
    Raises:
        HTTPException: If JWKS keys cannot be fetched
    """
    global _jwks_cache
    
    if not settings.oauth2_issuer_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OAuth2 issuer URL not configured"
        )
    
    # Use custom JWKS URL if provided, otherwise use standard path
    if settings.oauth2_jwks_url:
        jwks_url = settings.oauth2_jwks_url
    else:
        jwks_url = urljoin(settings.oauth2_issuer_url.rstrip('/') + '/', '.well-known/jwks.json')
    
    # Return cached keys if available (in production, you'd want TTL-based caching)
    if jwks_url in _jwks_cache:
        return _jwks_cache[jwks_url]
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(jwks_url)
            response.raise_for_status()
            jwks_data = response.json()
            
        # Cache the JWKS data
        _jwks_cache[jwks_url] = jwks_data
        logger.info(f"Fetched and cached JWKS keys from {jwks_url}")
        
        return jwks_data
        
    except Exception as e:
        logger.error(f"Failed to fetch JWKS keys from {jwks_url}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unable to fetch JWKS keys: {str(e)}"
        )


async def verify_jwt_token(token: str) -> Dict[str, Any]:
    """
    Verify and decode a JWT token.
    
    Args:
        token: The JWT token to verify
        
    Returns:
        Dict containing the decoded token payload
        
    Raises:
        HTTPException: If token is invalid
    """
    try:
        # Get JWKS keys for signature verification
        jwks_data = await get_jwks_keys()
        
        # Decode token header to get kid (key ID)
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get('kid')
        
        if not kid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing 'kid' in header"
            )
        
        # Find the matching key
        key = None
        for jwk in jwks_data.get('keys', []):
            if jwk.get('kid') == kid:
                key = jwk
                break
        
        if not key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Unable to find key with kid: {kid}"
            )
        
        # Verify and decode the token
        algorithms = ['RS256', 'ES256', 'HS256']  # Common algorithms
        payload = jwt.decode(
            token,
            key,
            algorithms=algorithms,
            audience=settings.oauth2_audience,
            issuer=settings.oauth2_issuer_url
        )
        
        logger.debug(f"Successfully verified JWT token for subject: {payload.get('sub', 'unknown')}")
        return payload
        
    except JWTError as e:
        logger.warning(f"JWT validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid JWT: {str(e)}"
        )
    except JWKError as e:
        logger.warning(f"JWK error during JWT validation: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"JWT key error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during JWT validation: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token validation failed: {str(e)}"
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(oauth2_scheme)
) -> Dict[str, Any]:
    """
    FastAPI dependency to get the current authenticated user.
    
    This function checks the DISABLE_AUTH setting first. If authentication is disabled,
    it returns a fake user for local development. Otherwise, it validates the JWT token.
    
    Args:
        credentials: HTTP Bearer token credentials
        
    Returns:
        Dict containing user information from the JWT payload
        
    Raises:
        HTTPException: If authentication fails
    """
    # Check if authentication is disabled for local development
    if settings.disable_auth:
        logger.debug("Authentication disabled, returning local dev user")
        return {
            "sub": "local-dev-user",
            "roles": ["admin"],
            "iss": "local-dev",
            "aud": "local-dev"
        }
    
    # Require valid credentials when auth is enabled
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify the JWT token
    return await verify_jwt_token(credentials.credentials)


def clear_jwks_cache():
    """Clear the JWKS cache. Useful for testing or key rotation."""
    global _jwks_cache
    _jwks_cache.clear()
    logger.info("JWKS cache cleared")