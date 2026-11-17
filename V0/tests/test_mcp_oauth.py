import threading
from unittest.mock import patch

import httpx
import pytest
from fastapi import HTTPException

from agent_memory_server.auth import UserInfo
from agent_memory_server.config import settings
from agent_memory_server.main import app
from agent_memory_server.mcp import JWTTokenVerifier, _build_mcp_auth_kwargs


@pytest.fixture
def oauth_settings(monkeypatch):
    monkeypatch.setattr(settings, "disable_auth", False)
    monkeypatch.setattr(settings, "auth_mode", "oauth2")
    monkeypatch.setattr(settings, "oauth2_issuer_url", "https://issuer.example")
    monkeypatch.setattr(settings, "oauth2_resource_host", "memory.example")


@pytest.mark.asyncio
async def test_resource_metadata(oauth_settings):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/.well-known/oauth-protected-resource")
        assert response.status_code == 200
        assert response.json() == {
            "resource": "https://memory.example",
            "authorization_servers": ["https://issuer.example"],
        }
        assert (
            await client.get("/.well-known/oauth-authorization-server")
        ).status_code == 404


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("disable_auth", True),
        ("auth_mode", "disabled"),
        ("auth_mode", "token"),
        ("oauth2_issuer_url", None),
        ("oauth2_resource_host", None),
    ],
)
@pytest.mark.asyncio
async def test_metadata_and_mcp_auth_disabled_together(
    oauth_settings, monkeypatch, field, value
):
    monkeypatch.setattr(settings, field, value)
    assert _build_mcp_auth_kwargs() == {}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (
            await client.get("/.well-known/oauth-protected-resource")
        ).status_code == 404


@pytest.mark.asyncio
async def test_verifier_runs_in_thread_and_returns_claims(oauth_settings):
    event_loop_thread = threading.get_ident()

    def verify(token):
        assert threading.get_ident() != event_loop_thread
        return UserInfo(sub="client", scope="read write", exp=123)

    with patch("agent_memory_server.mcp.verify_jwt", side_effect=verify):
        result = await JWTTokenVerifier().verify_token("token")
    assert result.client_id == "client"
    assert result.scopes == ["read", "write"]
    assert result.expires_at == 123
    assert _build_mcp_auth_kwargs()["token_verifier"] is not None


@pytest.mark.parametrize("status", [401, 500, 503])
@pytest.mark.asyncio
async def test_verifier_only_swallows_auth_failures(status):
    with patch(
        "agent_memory_server.mcp.verify_jwt",
        side_effect=HTTPException(status_code=status),
    ):
        if status == 401:
            assert await JWTTokenVerifier().verify_token("invalid") is None
        else:
            with pytest.raises(HTTPException) as error:
                await JWTTokenVerifier().verify_token("token")
            assert error.value.status_code == status


@pytest.mark.asyncio
async def test_verifier_propagates_unexpected_errors():
    with (
        patch(
            "agent_memory_server.mcp.verify_jwt", side_effect=RuntimeError("failure")
        ),
        pytest.raises(RuntimeError, match="failure"),
    ):
        await JWTTokenVerifier().verify_token("token")
