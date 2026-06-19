#!/bin/bash

# Quick Auth0 Setup Script for Redis Memory Server
# This script helps you quickly set up and test Auth0 authentication

set -e

echo "🔮 Redis Memory Server - Auth0 Quick Setup"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Please run this script from the redis-memory-server root directory"
    exit 1
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp manual_oauth_qa/env_template .env
    echo "✅ Created .env file"
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env and update the following values:"
    echo "   - OAUTH2_ISSUER_URL (your Auth0 domain)"
    echo "   - OAUTH2_AUDIENCE (your Auth0 API identifier)"
    echo "   - AUTH0_CLIENT_ID (your Auth0 application client ID)"
    echo "   - AUTH0_CLIENT_SECRET (your Auth0 application client secret)"
    echo "   - OPENAI_API_KEY (your OpenAI API key)"
    echo "   - ANTHROPIC_API_KEY (your Anthropic API key)"
    echo ""
    echo "Then run this script again to continue setup."
    exit 0
else
    echo "✅ .env file already exists"
fi

# Check Redis connection
echo "🔍 Checking Redis connection..."
if redis-cli ping >/dev/null 2>&1; then
    echo "✅ Redis is running"
else
    echo "❌ Redis is not running. Please start Redis:"
    echo "   brew services start redis  # macOS with Homebrew"
    echo "   sudo systemctl start redis # Linux with systemd"
    echo "   redis-server               # Manual start"
    exit 1
fi

# Check Auth0 configuration
echo "🔍 Checking Auth0 configuration..."
source .env

if [[ -z "$OAUTH2_ISSUER_URL" || "$OAUTH2_ISSUER_URL" == "https://your-domain.auth0.com/" ]]; then
    echo "❌ OAUTH2_ISSUER_URL not configured in .env"
    exit 1
fi

if [[ -z "$OAUTH2_AUDIENCE" || "$OAUTH2_AUDIENCE" == "https://api.redis-memory-server.com" ]]; then
    echo "❌ OAUTH2_AUDIENCE not configured in .env"
    exit 1
fi

if [[ -z "$AUTH0_CLIENT_ID" || "$AUTH0_CLIENT_ID" == "your-auth0-client-id" ]]; then
    echo "❌ AUTH0_CLIENT_ID not configured in .env"
    exit 1
fi

if [[ -z "$AUTH0_CLIENT_SECRET" || "$AUTH0_CLIENT_SECRET" == "your-auth0-client-secret" ]]; then
    echo "❌ AUTH0_CLIENT_SECRET not configured in .env"
    exit 1
fi

echo "✅ Auth0 configuration looks good"

# Test Auth0 token endpoint
echo "🔍 Testing Auth0 token endpoint..."
DOMAIN=$(echo $OAUTH2_ISSUER_URL | sed 's|https://||' | sed 's|/$||')
TOKEN_RESPONSE=$(curl -s -X POST "https://$DOMAIN/oauth/token" \
    -H "Content-Type: application/json" \
    -d "{
    \"client_id\": \"$AUTH0_CLIENT_ID\",
    \"client_secret\": \"$AUTH0_CLIENT_SECRET\",
    \"audience\": \"$OAUTH2_AUDIENCE\",
    \"grant_type\": \"client_credentials\"
  }")

if echo "$TOKEN_RESPONSE" | grep -q "access_token"; then
    echo "✅ Successfully obtained Auth0 token"
else
    echo "❌ Failed to get Auth0 token:"
    echo "$TOKEN_RESPONSE"
    exit 1
fi

# Check if memory server is running
echo "🔍 Checking memory server..."
PORT=${PORT:-8000}
if curl -s "http://localhost:$PORT/v1/health" >/dev/null 2>&1; then
    echo "✅ Memory server is running on port $PORT"

    echo ""
    echo "🚀 Setup complete! You can now:"
    echo ""
    echo "1. Run the comprehensive Auth0 test:"
    echo "   python manual_oauth_qa/test_auth0.py"
    echo ""
    echo "2. Run setup checks:"
    echo "   python manual_oauth_qa/setup_check.py"
    echo ""
    echo "3. Debug Auth0 configuration:"
    echo "   python manual_oauth_qa/debug_auth0.py"
    echo ""
    echo "🎉 Happy testing!"
else
    echo "❌ Memory server is not running on port $PORT"
    echo ""
    echo "Start the memory server with:"
    echo "   uv run python -m agent_memory_server.main"
    echo ""
    echo "Then run the Auth0 tests:"
    echo "   python manual_oauth_qa/test_auth0.py"
fi
