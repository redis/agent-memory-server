#!/usr/bin/env python3
import os

import httpx
from dotenv import load_dotenv


load_dotenv()

AUTH0_DOMAIN = (
    os.getenv("OAUTH2_ISSUER_URL", "").replace("https://", "").replace("/", "")
)
AUTH0_CLIENT_ID = os.getenv("AUTH0_CLIENT_ID")
AUTH0_CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET")
AUTH0_AUDIENCE = os.getenv("OAUTH2_AUDIENCE")

print("=== Auth0 Configuration Debug ===")
print(f"Domain: {AUTH0_DOMAIN}")
print(f"Client ID: {AUTH0_CLIENT_ID}")
print(
    f"Client Secret: {AUTH0_CLIENT_SECRET[:10]}..." if AUTH0_CLIENT_SECRET else "None"
)
print(f"Audience: {AUTH0_AUDIENCE}")
print(f"Token URL: https://{AUTH0_DOMAIN}/oauth/token")

print("\n=== Validation ===")
missing = []
if not AUTH0_DOMAIN:
    missing.append("OAUTH2_ISSUER_URL")
if not AUTH0_CLIENT_ID:
    missing.append("AUTH0_CLIENT_ID")
if not AUTH0_CLIENT_SECRET:
    missing.append("AUTH0_CLIENT_SECRET")
if not AUTH0_AUDIENCE:
    missing.append("OAUTH2_AUDIENCE")

if missing:
    print(f"❌ Missing: {', '.join(missing)}")
    exit(1)
else:
    print("✅ All required values present")

print("\n=== Testing Auth0 Token Request ===")
token_url = f"https://{AUTH0_DOMAIN}/oauth/token"
payload = {
    "client_id": AUTH0_CLIENT_ID,
    "client_secret": AUTH0_CLIENT_SECRET,
    "audience": AUTH0_AUDIENCE,
    "grant_type": "client_credentials",
}

print(f"Request URL: {token_url}")
print(f"Request payload: {payload}")

try:
    with httpx.Client(timeout=10.0) as client:
        response = client.post(
            token_url, json=payload, headers={"Content-Type": "application/json"}
        )
        print(f"Response status: {response.status_code}")
        print(f"Response: {response.text}")

        if response.status_code == 200:
            print("✅ Auth0 token request successful!")
        else:
            print("❌ Auth0 token request failed!")

except Exception as e:
    print(f"❌ Exception during token request: {e}")
