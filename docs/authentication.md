# Authentication

The Redis Agent Memory Server supports OAuth2/JWT Bearer token authentication for secure API access. All API endpoints (except `/health`, `/docs`, and `/openapi.json`) require valid JWT authentication unless disabled for development.

## Features

- **OAuth2/JWT Bearer Token Authentication**: Industry-standard authentication using JWT access tokens
- **JWKS Public Key Validation**: Automatic fetching and caching of public keys for token signature verification
- **Multi-Provider Support**: Compatible with Auth0, AWS Cognito, Okta, Azure AD, and any standard OAuth2 provider
- **Flexible Configuration**: Environment variable-based configuration for different deployment scenarios
- **Development Mode**: `DISABLE_AUTH` setting for local development and testing
- **Role and Scope Support**: Fine-grained access control using JWT claims

## Configuration

Authentication is configured using environment variables:

```bash
# OAuth2 Provider Configuration
OAUTH2_ISSUER_URL=https://your-auth-provider.com
OAUTH2_AUDIENCE=your-api-audience
OAUTH2_JWKS_URL=https://your-auth-provider.com/.well-known/jwks.json  # Optional, auto-derived from issuer
OAUTH2_ALGORITHMS=["RS256"]  # Supported signing algorithms

# Development Mode (DISABLE AUTHENTICATION - USE ONLY FOR DEVELOPMENT)
DISABLE_AUTH=true  # Set to true to bypass all authentication (development only)
```

## Provider Examples

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

### With Authentication (Production)

```bash
# Make authenticated API request
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     -H "Content-Type: application/json" \
     http://localhost:8000/sessions/

# Python example
import httpx

headers = {
    "Authorization": "Bearer YOUR_JWT_TOKEN",
    "Content-Type": "application/json"
}

response = httpx.get("http://localhost:8000/sessions/", headers=headers)
```

### Development Mode (Local Testing)

```bash
# Set environment variable to disable auth
export DISABLE_AUTH=true

# Now you can make requests without tokens
curl -H "Content-Type: application/json" \
     http://localhost:8000/sessions/
```

## Token Requirements

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

- `Missing authorization header`: No `Authorization: Bearer` header provided
- `Missing bearer token`: Empty or malformed authorization header
- `Invalid token header`: Malformed JWT structure
- `Token has expired`: JWT `exp` claim is in the past
- `Invalid audience`: JWT `aud` claim doesn't match expected audience
- `Unable to find matching public key`: JWKS doesn't contain key for token's `kid`

## Security Best Practices

1. **Never use `DISABLE_AUTH=true` in production**
2. **Use HTTPS in production** to protect tokens in transit
3. **Implement token refresh** in your clients for long-running applications
4. **Monitor token expiration** and handle 401 responses appropriately
5. **Validate tokens server-side** - never trust client-side validation alone
6. **Use appropriate scopes/roles** for fine-grained access control

## Manual Testing

For comprehensive Auth0 testing instructions, see the [manual OAuth testing guide](../manual_oauth_qa/README.md).
