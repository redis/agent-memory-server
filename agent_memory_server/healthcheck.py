import time

from fastapi import APIRouter, Depends

from agent_memory_server.auth import UserInfo, get_current_user
from agent_memory_server.fips import get_fips_diagnostics
from agent_memory_server.models import HealthCheckResponse


router = APIRouter()


@router.get("/v1/health", response_model=HealthCheckResponse)
async def get_health():
    """
    Health check endpoint

    Returns:
        HealthCheckResponse with current timestamp
    """
    # Return current time in milliseconds
    return HealthCheckResponse(now=int(time.time() * 1000))


@router.get("/v1/fips")
async def get_fips_status(
    user: UserInfo = Depends(get_current_user),
) -> dict:
    """
    FIPS compliance diagnostics endpoint (authenticated).

    Returns runtime FIPS posture including OpenSSL version,
    kernel FIPS mode, and configuration status.
    """
    return get_fips_diagnostics()
