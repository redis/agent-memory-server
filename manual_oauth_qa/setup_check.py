#!/usr/bin/env python3
"""
Setup Check Script for Auth0 Manual Testing

This script verifies that all dependencies and configuration
are properly set up for Auth0 testing.
"""

import os
import sys
from pathlib import Path


def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    print(
        f"❌ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.8+"
    )
    return False


def check_dependencies():
    """Check required Python packages"""
    print("\n📦 Checking Python dependencies...")
    required_packages = [
        "httpx",
        "structlog",
        "python-dotenv",
        "fastapi",
        "uvicorn",
        "redis",
        "pydantic",
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing.append(package)

    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    print("✅ All required packages installed")
    return True


def check_redis():
    """Check Redis connection"""
    print("\n🔴 Checking Redis connection...")
    try:
        import redis

        r = redis.Redis(host="localhost", port=6379, decode_responses=True)
        r.ping()
        print("✅ Redis is running and accessible")
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("Start Redis with: redis-server")
        return False


def check_env_file():
    """Check if .env file exists and has required variables"""
    print("\n📄 Checking .env file...")
    env_path = Path(".env")

    if not env_path.exists():
        print("❌ .env file not found")
        print("Copy env_template to .env and configure it:")
        print("cp manual_oauth_qa/env_template .env")
        return False

    print("✅ .env file exists")

    # Check required variables
    from dotenv import load_dotenv

    load_dotenv()

    required_vars = [
        "OAUTH2_ISSUER_URL",
        "OAUTH2_AUDIENCE",
        "AUTH0_CLIENT_ID",
        "AUTH0_CLIENT_SECRET",
    ]

    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if not value or value.startswith("your-"):
            missing_vars.append(var)
            print(f"❌ {var} - Not configured")
        else:
            print(f"✅ {var} - Configured")

    if missing_vars:
        print(f"\n❌ Configure these variables in .env: {', '.join(missing_vars)}")
        return False
    print("✅ All required environment variables configured")
    return True


def check_memory_server():
    """Check if memory server is accessible"""
    print("\n🧠 Checking memory server...")
    try:
        import httpx

        port = os.getenv("PORT", "8000")
        response = httpx.get(f"http://localhost:{port}/health", timeout=5.0)
        if response.status_code == 200:
            print(f"✅ Memory server running on port {port}")
            return True
        print(f"❌ Memory server responded with status {response.status_code}")
        return False
    except Exception as e:
        print(f"❌ Memory server not accessible: {e}")
        print("Start the server with: uv run python -m agent_memory_server.main")
        return False


def main():
    """Run all checks"""
    print("🔍 Redis Memory Server - Auth0 Setup Check")
    print("=" * 50)

    checks = [
        check_python_version(),
        check_dependencies(),
        check_redis(),
        check_env_file(),
        check_memory_server(),
    ]

    passed = sum(checks)
    total = len(checks)

    print(f"\n📊 Setup Check Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All checks passed! Ready for Auth0 testing.")
        print("\nNext steps:")
        print("1. Run: python manual_oauth_qa/test_auth0.py")
        return True
    print("❌ Some checks failed. Please fix the issues above.")
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
