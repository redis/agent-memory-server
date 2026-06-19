#!/usr/bin/env python3
"""
Manual Auth0 Testing Script for Redis Memory Server

This script helps you test Auth0 authentication with the Redis Memory Server.
It will:
1. Get an access token from Auth0
2. Test various API endpoints with the token
3. Verify authentication is working correctly

Prerequisites:
1. Auth0 application configured (Machine to Machine)
2. .env file with Auth0 configuration
3. Redis server running
4. Memory server running with authentication enabled
"""

import os
import sys
import time
from typing import Any

import httpx
import structlog
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Configure logging
logger = structlog.get_logger()

# Auth0 Configuration
AUTH0_DOMAIN = (
    os.getenv("OAUTH2_ISSUER_URL", "").replace("https://", "").replace("/", "")
)
AUTH0_CLIENT_ID = os.getenv("AUTH0_CLIENT_ID")
AUTH0_CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET")
AUTH0_AUDIENCE = os.getenv("OAUTH2_AUDIENCE")

# Memory Server Configuration
MEMORY_SERVER_URL = f"http://localhost:{os.getenv('PORT', '8000')}"


class Auth0Tester:
    def __init__(self):
        self.access_token = None
        self.client = httpx.Client(timeout=30.0)

    def get_auth0_token(self) -> str:
        """Get an access token from Auth0"""
        if not all(
            [AUTH0_DOMAIN, AUTH0_CLIENT_ID, AUTH0_CLIENT_SECRET, AUTH0_AUDIENCE]
        ):
            raise ValueError(
                "Missing Auth0 configuration. Please set:\n"
                "- OAUTH2_ISSUER_URL (e.g., https://your-domain.auth0.com/)\n"
                "- AUTH0_CLIENT_ID\n"
                "- AUTH0_CLIENT_SECRET\n"
                "- OAUTH2_AUDIENCE"
            )

        token_url = f"https://{AUTH0_DOMAIN}/oauth/token"

        payload = {
            "client_id": AUTH0_CLIENT_ID,
            "client_secret": AUTH0_CLIENT_SECRET,
            "audience": AUTH0_AUDIENCE,
            "grant_type": "client_credentials",
        }

        headers = {"Content-Type": "application/json"}

        logger.info(
            "Requesting Auth0 access token",
            domain=AUTH0_DOMAIN,
            audience=AUTH0_AUDIENCE,
        )

        try:
            response = self.client.post(token_url, json=payload, headers=headers)
            response.raise_for_status()

            token_data = response.json()
            self.access_token = token_data["access_token"]

            logger.info(
                "Successfully obtained Auth0 token",
                token_type=token_data.get("token_type"),
                expires_in=token_data.get("expires_in"),
            )

            return self.access_token

        except httpx.HTTPError as e:
            logger.error("Failed to get Auth0 token", error=str(e))
            if hasattr(e, "response") and e.response:
                logger.error("Auth0 error response", response=e.response.text)
            raise

    def test_endpoint(
        self, method: str, endpoint: str, data: dict[str, Any] = None
    ) -> dict[str, Any]:
        """Test a memory server endpoint with authentication"""
        url = f"{MEMORY_SERVER_URL}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

        logger.info(f"Testing {method} {endpoint}")

        try:
            if method.upper() == "GET":
                response = self.client.get(url, headers=headers)
            elif method.upper() == "POST":
                response = self.client.post(url, headers=headers, json=data or {})
            elif method.upper() == "PUT":
                response = self.client.put(url, headers=headers, json=data or {})
            elif method.upper() == "DELETE":
                response = self.client.delete(url, headers=headers)
            else:
                raise ValueError(f"Unsupported method: {method}")

            result = {
                "status_code": response.status_code,
                "success": response.status_code < 400,
                "response": response.json()
                if response.headers.get("content-type", "").startswith(
                    "application/json"
                )
                else response.text,
            }

            if result["success"]:
                logger.info(
                    f"✅ {method} {endpoint} - Success", status=response.status_code
                )
            else:
                logger.error(
                    f"❌ {method} {endpoint} - Failed",
                    status=response.status_code,
                    response=result["response"],
                )

            return result

        except Exception as e:
            logger.error(f"❌ {method} {endpoint} - Exception", error=str(e))
            return {"status_code": 0, "success": False, "error": str(e)}

    def run_comprehensive_test(self):
        """Run a comprehensive test of all endpoints"""
        logger.info("🚀 Starting comprehensive Auth0 authentication test")

        # Step 1: Get Auth0 token
        try:
            self.get_auth0_token()
        except Exception as e:
            logger.error("Failed to get Auth0 token, aborting tests", error=str(e))
            return False

        # Step 2: Test health endpoint (should work without auth)
        logger.info("\n📋 Testing health endpoint (no auth required)")
        health_result = self.test_endpoint("GET", "/v1/health")

        # Step 3: Test authenticated endpoints
        logger.info("\n🔐 Testing authenticated endpoints")

        test_cases = [
            # Sessions endpoints
            ("GET", "/sessions/", None),
            # Memory endpoints
            (
                "POST",
                "/memory-prompt",
                {
                    "query": "What is the capital of France?",
                    "session": {
                        "session_id": "test-session-auth0",
                        "namespace": "test-auth0",
                        "model_name": "gpt-4o-mini",
                    },
                },
            ),
            (
                "POST",
                "/long-term-memory",
                {
                    "memories": [
                        {
                            "id": "auth0-test-memory-1",
                            "text": "Auth0 test memory",
                            "session_id": "test-session-auth0",
                            "namespace": "test-auth0",
                        }
                    ]
                },
            ),
            ("POST", "/long-term-memory/search", {"text": "Auth0 test", "limit": 5}),
        ]

        results = []
        for method, endpoint, data in test_cases:
            result = self.test_endpoint(method, endpoint, data)
            results.append((method, endpoint, result))
            time.sleep(0.5)  # Small delay between requests

        # Step 4: Test without token (should fail)
        logger.info("\n🚫 Testing without authentication (should fail)")
        old_token = self.access_token
        self.access_token = None

        no_auth_result = self.test_endpoint("GET", "/sessions/")
        expected_failure = no_auth_result["status_code"] == 401

        if expected_failure:
            logger.info("✅ Correctly rejected request without authentication")
        else:
            logger.error(
                "❌ Request without authentication should have failed with 401"
            )

        # Restore token
        self.access_token = old_token

        # Step 5: Test with invalid token (should fail)
        logger.info("\n🚫 Testing with invalid token (should fail)")
        self.access_token = "invalid.jwt.token"

        invalid_token_result = self.test_endpoint("GET", "/sessions/")
        expected_invalid_failure = invalid_token_result["status_code"] == 401

        if expected_invalid_failure:
            logger.info("✅ Correctly rejected request with invalid token")
        else:
            logger.error("❌ Request with invalid token should have failed with 401")

        # Restore token
        self.access_token = old_token

        # Step 6: Summary
        logger.info("\n📊 Test Summary")
        successful_tests = sum(1 for _, _, result in results if result["success"])
        total_tests = len(results)

        logger.info(f"Authenticated endpoints: {successful_tests}/{total_tests} passed")
        logger.info(f"Health endpoint: {'✅' if health_result['success'] else '❌'}")
        logger.info(f"No auth rejection: {'✅' if expected_failure else '❌'}")
        logger.info(
            f"Invalid token rejection: {'✅' if expected_invalid_failure else '❌'}"
        )

        overall_success = (
            successful_tests == total_tests
            and health_result["success"]
            and expected_failure
            and expected_invalid_failure
        )

        if overall_success:
            logger.info("🎉 All Auth0 authentication tests passed!")
        else:
            logger.error("❌ Some Auth0 authentication tests failed")

        return overall_success


def main():
    """Main function to run Auth0 tests"""
    print("🔮 Redis Memory Server - Auth0 Manual Testing")
    print("=" * 50)

    # Check if memory server is running
    try:
        response = httpx.get(f"{MEMORY_SERVER_URL}/v1/health", timeout=5.0)
        if response.status_code != 200:
            print(f"❌ Memory server not responding correctly at {MEMORY_SERVER_URL}")
            print("Please start the memory server first:")
            print("  uv run python -m agent_memory_server.main")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Cannot connect to memory server at {MEMORY_SERVER_URL}")
        print(f"Error: {e}")
        print("Please start the memory server first:")
        print("  uv run python -m agent_memory_server.main")
        sys.exit(1)

    print(f"✅ Memory server is running at {MEMORY_SERVER_URL}")

    # Run tests
    tester = Auth0Tester()
    success = tester.run_comprehensive_test()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
