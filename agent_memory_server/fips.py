"""FIPS 140-3 compliance diagnostics."""

import hashlib
import ssl


def get_fips_diagnostics() -> dict:
    """Check runtime FIPS compliance posture."""
    from agent_memory_server.config import settings

    diagnostics: dict = {}

    # OpenSSL version
    diagnostics["openssl_version"] = ssl.OPENSSL_VERSION

    # Kernel FIPS mode (Linux only)
    try:
        with open("/proc/sys/crypto/fips_enabled") as f:
            diagnostics["kernel_fips_enabled"] = f.read().strip() == "1"
    except FileNotFoundError:
        diagnostics["kernel_fips_enabled"] = None  # Not Linux or not available

    # Test if non-FIPS algorithms are blocked
    try:
        hashlib.md5(b"test")
        diagnostics["md5_blocked"] = False
    except ValueError:
        diagnostics["md5_blocked"] = True

    # Token hash algorithm
    diagnostics["token_hash_algorithm"] = settings.token_hash_algorithm

    # Redis TLS
    diagnostics["redis_tls_enabled"] = settings.redis_url.startswith("rediss://")

    # Overall assessment
    diagnostics["fips_capable"] = (
        diagnostics.get("md5_blocked", False)
        and settings.token_hash_algorithm != "bcrypt"
    )

    return diagnostics
