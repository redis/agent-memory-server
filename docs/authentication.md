# Authentication

The Redis Agent Memory Server supports multiple authentication modes for secure API access. All API endpoints (except `/health`, `/docs`, and `/openapi.json`) require valid authentication unless disabled for development.

## Authentication Modes

The server supports three authentication modes:

1. **Disabled** (default): No authentication required - suitable for development only
2. **Token Authentication**: Simple API tokens stored in Redis with optional expiration
3. **OAuth2/JWT**: Industry-standard authentication using JWT access tokens

## Features

- **Simple Token Authentication**: Generate and manage API tokens via CLI with optional expiration
- **OAuth2/JWT Bearer Token Authentication**: Industry-standard authentication using JWT access tokens
- **JWKS Public Key Validation**: Automatic fetching and caching of public keys for token signature verification
- **Multi-Provider Support**: Compatible with Auth0, AWS Cognito, Okta, Azure AD, and any standard OAuth2 provider
- **Flexible Configuration**: Environment variable-based configuration for different deployment scenarios
- **Development Mode**: `DISABLE_AUTH` setting for local development and testing
- **Role and Scope Support**: Fine-grained access control using JWT claims
- **CLI Token Management**: Create, list, view, and remove tokens via command line

## Configuration

### Basic Configuration

Authentication is configured using environment variables:

```bash
# Authentication Mode Selection
AUTH_MODE=disabled  # Options: disabled, token, oauth2 (default: disabled)
# OR legacy setting:
DISABLE_AUTH=true   # Set to true to bypass all authentication (development only)

# Token Authentication (when AUTH_MODE=token)
TOKEN_AUTH_ENABLED=true  # Alternative way to enable token auth

# OAuth2 Provider Configuration (when AUTH_MODE=oauth2)
OAUTH2_ISSUER_URL=https://your-auth-provider.com
OAUTH2_AUDIENCE=your-api-audience
OAUTH2_JWKS_URL=https://your-auth-provider.com/.well-known/jwks.json  # Optional, auto-derived from issuer
OAUTH2_ALGORITHMS=["RS256"]  # Supported signing algorithms
```

### Token Authentication Setup

To use token authentication:

1. **Enable token authentication:**
   ```bash
   export AUTH_MODE=token
   # OR
   export TOKEN_AUTH_ENABLED=true
   ```

2. **Create tokens using the CLI:**
   ```bash
   # Create a token with 30-day expiration
   uv run agent-memory token add --description "API access token" --expires-days 30

   # Create a permanent token (no expiration)
   uv run agent-memory token add --description "Service account token"
   ```

3. **Use the token in API requests:**
   ```bash
   curl -H "Authorization: Bearer YOUR_TOKEN" \
        http://localhost:8000/v1/working-memory/
   ```

### Token Management Commands

The CLI provides comprehensive token management:

```bash
# List all tokens (shows masked token hashes)
uv run agent-memory token list

# Show details for a specific token (supports partial hash matching)
uv run agent-memory token show abc12345

# Remove a token (with confirmation)
uv run agent-memory token remove abc12345

# Remove a token without confirmation
uv run agent-memory token remove abc12345 --force
```

**Security Features:**
- Tokens are hashed using bcrypt before storage
- Only hashed values are stored in Redis (server never has access to plaintext tokens)
- Automatic expiration using Redis TTL
- Secure token generation using `secrets.token_urlsafe()`

## OAuth2 Provider Examples

### Auth0

```bash
OAUTH2_ISSUER_URL=https://your-domain.auth0.com/
OAUTH2_AUDIENCE=https://your-api.com
```

### AWS Cognito

```bash
OAUTH2_ISSUER_URL=https://cognito-idp.region.amazonaws.com/your-user-pool-id
OAUTH2_AUDIENCE=your-app-client-id
```

### Okta

```bash
OAUTH2_ISSUER_URL=https://your-domain.okta.com/oauth2/default
OAUTH2_AUDIENCE=api://default
```

### Azure AD

```bash
OAUTH2_ISSUER_URL=https://login.microsoftonline.com/your-tenant-id/v2.0
OAUTH2_AUDIENCE=your-application-id
```

## Usage Examples

### With Token Authentication

```bash
# First, create a token
uv run agent-memory token add --description "My API token" --expires-days 30

# Use the returned token in API requests
curl -H "Authorization: Bearer YOUR_API_TOKEN" \
     -H "Content-Type: application/json" \
     http://localhost:8000/v1/working-memory/

# Python example
import httpx

headers = {
    "Authorization": "Bearer YOUR_API_TOKEN",
    "Content-Type": "application/json"
}

response = httpx.get("http://localhost:8000/v1/working-memory/", headers=headers)
```

### With OAuth2/JWT Authentication

```bash
# Make authenticated API request with JWT
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     -H "Content-Type: application/json" \
     http://localhost:8000/v1/working-memory/

# Python example
import httpx

headers = {
    "Authorization": "Bearer YOUR_JWT_TOKEN",
    "Content-Type": "application/json"
}

response = httpx.get("http://localhost:8000/v1/working-memory/", headers=headers)
```

### Development Mode (Local Testing)

```bash
# Set environment variable to disable auth
export DISABLE_AUTH=true

# Now you can make requests without tokens
curl -H "Content-Type: application/json" \
     http://localhost:8000/v1/working-memory/
```

## Token Requirements

### API Tokens (Token Authentication)

- **Valid token**: Must exist in Redis and not be expired
- **Secure generation**: Generated using cryptographically secure random bytes
- **Optional expiration**: Tokens can be created with or without expiration dates

### JWT Tokens (OAuth2 Authentication)

JWT tokens must include:

- **Valid signature**: Verified using JWKS public keys from the issuer
- **Not expired**: `exp` claim must be in the future
- **Valid audience**: `aud` claim must match `OAUTH2_AUDIENCE` (if configured)
- **Valid issuer**: `iss` claim must match `OAUTH2_ISSUER_URL`
- **Subject**: `sub` claim identifying the user/client

## Error Responses

Authentication failures return HTTP 401 with details:

```json
{
  "detail": "Invalid JWT: Token has expired",
  "status_code": 401
}
```

Common error scenarios:

**General Authentication Errors:**
- `Missing authorization header`: No `Authorization: Bearer` header provided
- `Missing bearer token`: Empty or malformed authorization header

**Token Authentication Errors:**
- `Invalid token`: Token not found in Redis or malformed
- `Token has expired`: Token expiration date has passed

**OAuth2/JWT Authentication Errors:**
- `Invalid token header`: Malformed JWT structure
- `Token has expired`: JWT `exp` claim is in the past
- `Invalid audience`: JWT `aud` claim doesn't match expected audience
- `Unable to find matching public key`: JWKS doesn't contain key for token's `kid`

## Security Best Practices

### General Security

1. **Never use `DISABLE_AUTH=true` in production**
2. **Use HTTPS in production** to protect tokens in transit
3. **Monitor authentication failures** and implement rate limiting if needed
4. **Handle 401 responses appropriately** in your clients
5. **Validate tokens server-side** - never trust client-side validation alone

### Token Authentication Security

6. **Use token expiration** when creating tokens for enhanced security
7. **Rotate tokens regularly** by removing old tokens and creating new ones
8. **Store tokens securely** in your applications (use environment variables, not code)
9. **Remove unused tokens** using the CLI to minimize attack surface
10. **Monitor token usage** and remove tokens that are no longer needed

### OAuth2/JWT Security

11. **Implement token refresh** in your clients for long-running applications
12. **Use appropriate scopes/roles** for fine-grained access control
13. **Validate token expiration** and audience claims properly
14. **Cache JWKS appropriately** but refresh periodically

## Manual Testing

For comprehensive Auth0 testing instructions, see the [manual OAuth testing guide](https://github.com/redis/agent-memory-server/tree/main/manual_oauth_qa/README.md).
