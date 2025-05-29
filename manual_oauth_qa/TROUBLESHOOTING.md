# Auth0 Troubleshooting Guide

This guide helps you diagnose and fix common Auth0 authentication issues.

## 🚨 Common Error Messages

### 403 Forbidden - "access_denied"

**Error**: `Service not enabled within domain: https://your-api.com`

**Cause**: Your Auth0 application is not authorized for the specified audience.

**Solution**:
1. Go to Auth0 Dashboard → **APIs** → Your API
2. Click **Machine to Machine Applications** tab
3. Find your application and toggle it **ON**
4. Save the changes

### 401 Unauthorized - "Invalid token header"

**Error**: `Invalid token header`

**Cause**: Token is malformed or missing.

**Solutions**:
1. Check that `Authorization: Bearer <token>` header is included
2. Verify the token is not expired
3. Ensure no extra spaces or characters in the token

### 401 Unauthorized - "Invalid audience"

**Error**: `Invalid audience. Expected: https://your-api.com`

**Cause**: Token audience doesn't match server configuration.

**Solution**:
1. Check `OAUTH2_AUDIENCE` in your `.env` file
2. Ensure it matches your Auth0 API identifier exactly
3. Verify your Auth0 application is authorized for this API

### Connection Refused

**Error**: `Connection refused` when testing endpoints

**Cause**: Memory server is not running.

**Solution**:
```bash
# Start the memory server
uv run python -m agent_memory_server.main
```

## 🔧 Debugging Steps

### Step 1: Check Environment Variables

```bash
# Check for conflicting environment variables
env | grep -E "(OAUTH2|AUTH0)"

# If you see conflicting values, unset them:
unset OAUTH2_AUDIENCE
unset OAUTH2_ISSUER_URL
```

### Step 2: Verify Auth0 Configuration

```bash
# Run the debug script
python manual_oauth_qa/debug_auth0.py
```

Expected output:
```
✅ All required values present
✅ Auth0 token request successful!
```

### Step 3: Test Auth0 Token Manually

```bash
# Replace with your actual values
curl -X POST "https://your-domain.auth0.com/oauth/token" \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "your-client-id",
    "client_secret": "your-client-secret",
    "audience": "https://api.redis-memory-server.com",
    "grant_type": "client_credentials"
  }'
```

### Step 4: Check Server Logs

Look for these messages when starting the server:

```
✅ Good: OAuth2 authentication configured
❌ Bad: Authentication is DISABLED
❌ Bad: OAuth2 issuer URL not configured
```

### Step 5: Verify API Authorization

1. Go to Auth0 Dashboard
2. Navigate to **APIs** → Your API → **Machine to Machine Applications**
3. Ensure your application is **authorized** (toggle ON)
4. Check that required scopes are selected

## 🐛 Specific Issues

### Issue: "Service not enabled within domain"

This happens when using the wrong audience or unauthorized application.

**Fix**:
1. Create a custom API in Auth0 (not Management API)
2. Use the custom API identifier as your audience
3. Authorize your application for the custom API

### Issue: Environment Variables Not Loading

**Symptoms**: Configuration looks correct but still fails

**Fix**:
```bash
# Check if shell environment is overriding .env
env | grep OAUTH2_AUDIENCE

# If found, unset it
unset OAUTH2_AUDIENCE

# Restart your shell or reload .env
source .env
```

### Issue: Token Expires Immediately

**Symptoms**: Token works once then fails

**Cause**: System clock is wrong or token caching issue

**Fix**:
1. Check system time: `date`
2. Sync time if needed
3. Clear any token caches

### Issue: JWKS Key Not Found

**Error**: `Unable to find matching public key for kid: xyz`

**Cause**: Auth0 key rotation or network issues

**Fix**:
1. Wait a few minutes for key propagation
2. Check network connectivity to Auth0
3. Verify JWKS URL is accessible

## 📋 Checklist for New Machine Setup

- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Redis server running (`redis-server`)
- [ ] `.env` file created and configured
- [ ] Auth0 application created and configured
- [ ] Auth0 API created and application authorized
- [ ] No conflicting environment variables
- [ ] Memory server starts without errors
- [ ] Auth0 token request succeeds

## 🆘 Getting Help

If you're still having issues:

1. **Run diagnostics**:
   ```bash
   python manual_oauth_qa/setup_check.py
   python manual_oauth_qa/debug_auth0.py
   ```

2. **Check Auth0 logs**:
   - Go to Auth0 Dashboard → **Monitoring** → **Logs**
   - Look for failed authentication attempts

3. **Verify network connectivity**:
   ```bash
   curl -I https://your-domain.auth0.com/.well-known/jwks.json
   ```

4. **Test with minimal example**:
   Use the debug script to isolate the issue

## 📞 Support Resources

- [Auth0 Documentation](https://auth0.com/docs)
- [Auth0 Community](https://community.auth0.com/)
- [Auth0 Support](https://support.auth0.com/)
- [JWT.io](https://jwt.io/) - For debugging JWT tokens
