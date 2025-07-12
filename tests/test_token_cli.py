"""Integration tests for token CLI commands."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from agent_memory_server.auth import TokenInfo
from agent_memory_server.cli import token


@pytest.fixture
def mock_redis():
    """Mock Redis connection for CLI tests."""
    return AsyncMock()


@pytest.fixture
def cli_runner():
    """Click CLI runner for testing."""
    return CliRunner()


class TestTokenCLI:
    """Test token CLI commands."""

    @patch("agent_memory_server.cli.get_redis_conn")
    def test_token_add_command(self, mock_get_redis, mock_redis, cli_runner):
        """Test token add command."""
        mock_get_redis.return_value = mock_redis

        result = cli_runner.invoke(
            token, ["add", "--description", "Test token", "--expires-days", "30"]
        )

        assert result.exit_code == 0
        assert "Token created successfully!" in result.output
        assert "Description: Test token" in result.output
        assert "WARNING: Save this token securely" in result.output

        # Verify Redis calls
        mock_redis.set.assert_called_once()
        mock_redis.expire.assert_called_once()
        mock_redis.sadd.assert_called_once()

    @patch("agent_memory_server.cli.get_redis_conn")
    def test_token_add_command_no_expiry(self, mock_get_redis, mock_redis, cli_runner):
        """Test token add command without expiration."""
        mock_get_redis.return_value = mock_redis

        result = cli_runner.invoke(token, ["add", "--description", "Permanent token"])

        assert result.exit_code == 0
        assert "Token created successfully!" in result.output
        assert "Expires: Never" in result.output

        # Verify Redis calls
        mock_redis.set.assert_called_once()
        mock_redis.expire.assert_not_called()  # No expiry set
        mock_redis.sadd.assert_called_once()

    @patch("agent_memory_server.cli.get_redis_conn")
    def test_token_list_command_empty(self, mock_get_redis, mock_redis, cli_runner):
        """Test token list command with no tokens."""
        mock_get_redis.return_value = mock_redis
        mock_redis.smembers.return_value = set()

        result = cli_runner.invoke(token, ["list"])

        assert result.exit_code == 0
        assert "No tokens found." in result.output

    @patch("agent_memory_server.cli.get_redis_conn")
    def test_token_list_command_with_tokens(
        self, mock_get_redis, mock_redis, cli_runner
    ):
        """Test token list command with tokens."""
        mock_get_redis.return_value = mock_redis

        # Create sample token data
        token_hash = "test_hash_123456789012345678901234567890"
        token_info = TokenInfo(
            description="Test token",
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(days=30),
            token_hash=token_hash,
        )

        mock_redis.smembers.return_value = {token_hash}
        mock_redis.get.return_value = token_info.model_dump_json()

        result = cli_runner.invoke(token, ["list"])

        assert result.exit_code == 0
        assert "Authentication Tokens:" in result.output
        assert "Test token" in result.output
        assert "test_has...34567890" in result.output  # Masked hash

    @patch("agent_memory_server.cli.get_redis_conn")
    def test_token_show_command(self, mock_get_redis, mock_redis, cli_runner):
        """Test token show command."""
        mock_get_redis.return_value = mock_redis

        # Create sample token data
        token_hash = "test_hash_123456789012345678901234567890"
        token_info = TokenInfo(
            description="Test token",
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(days=30),
            token_hash=token_hash,
        )

        mock_redis.get.return_value = token_info.model_dump_json()

        result = cli_runner.invoke(token, ["show", token_hash])

        assert result.exit_code == 0
        assert "Token Details:" in result.output
        assert "Test token" in result.output
        assert "Status: Active" in result.output

    @patch("agent_memory_server.cli.get_redis_conn")
    def test_token_show_command_partial_hash(
        self, mock_get_redis, mock_redis, cli_runner
    ):
        """Test token show command with partial hash."""
        mock_get_redis.return_value = mock_redis

        # Create sample token data
        token_hash = "test_hash_123456789012345678901234567890"
        token_info = TokenInfo(
            description="Test token",
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(days=30),
            token_hash=token_hash,
        )

        mock_redis.smembers.return_value = {token_hash}
        mock_redis.get.return_value = token_info.model_dump_json()

        result = cli_runner.invoke(token, ["show", "test_hash"])

        assert result.exit_code == 0
        assert "Token Details:" in result.output
        assert "Test token" in result.output

    @patch("agent_memory_server.cli.get_redis_conn")
    def test_token_show_command_not_found(self, mock_get_redis, mock_redis, cli_runner):
        """Test token show command with non-existent token."""
        mock_get_redis.return_value = mock_redis
        mock_redis.get.return_value = None

        result = cli_runner.invoke(token, ["show", "nonexistent"])

        assert result.exit_code == 0
        assert "No token found matching" in result.output

    @patch("agent_memory_server.cli.get_redis_conn")
    def test_token_remove_command_with_confirmation(
        self, mock_get_redis, mock_redis, cli_runner
    ):
        """Test token remove command with confirmation."""
        mock_get_redis.return_value = mock_redis

        # Create sample token data
        token_hash = "test_hash_123456789012345678901234567890"
        token_info = TokenInfo(
            description="Test token",
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(days=30),
            token_hash=token_hash,
        )

        mock_redis.get.return_value = token_info.model_dump_json()

        # Simulate user confirming removal
        result = cli_runner.invoke(token, ["remove", token_hash], input="y\n")

        assert result.exit_code == 0
        assert "Token to remove:" in result.output
        assert "Token removed successfully." in result.output

        # Verify Redis calls
        mock_redis.delete.assert_called_once()
        mock_redis.srem.assert_called_once()

    @patch("agent_memory_server.cli.get_redis_conn")
    def test_token_remove_command_force(self, mock_get_redis, mock_redis, cli_runner):
        """Test token remove command with force flag."""
        mock_get_redis.return_value = mock_redis

        # Create sample token data
        token_hash = "test_hash_123456789012345678901234567890"
        token_info = TokenInfo(
            description="Test token",
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(days=30),
            token_hash=token_hash,
        )

        mock_redis.get.return_value = token_info.model_dump_json()

        result = cli_runner.invoke(token, ["remove", token_hash, "--force"])

        assert result.exit_code == 0
        assert "Token removed successfully." in result.output

        # Verify Redis calls
        mock_redis.delete.assert_called_once()
        mock_redis.srem.assert_called_once()

    @patch("agent_memory_server.cli.get_redis_conn")
    def test_token_remove_command_cancelled(
        self, mock_get_redis, mock_redis, cli_runner
    ):
        """Test token remove command cancelled by user."""
        mock_get_redis.return_value = mock_redis

        # Create sample token data
        token_hash = "test_hash_123456789012345678901234567890"
        token_info = TokenInfo(
            description="Test token",
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(days=30),
            token_hash=token_hash,
        )

        mock_redis.get.return_value = token_info.model_dump_json()

        # Simulate user cancelling removal
        result = cli_runner.invoke(token, ["remove", token_hash], input="n\n")

        assert result.exit_code == 0
        assert "Token removal cancelled." in result.output

        # Verify Redis calls - should not delete
        mock_redis.delete.assert_not_called()
        mock_redis.srem.assert_not_called()

    @patch("agent_memory_server.cli.get_redis_conn")
    def test_token_remove_command_partial_hash_multiple_matches(
        self, mock_get_redis, mock_redis, cli_runner
    ):
        """Test token remove command with partial hash that matches multiple tokens."""
        mock_get_redis.return_value = mock_redis

        # Create multiple token hashes with same prefix
        token_hashes = {
            "test_hash_111111111111111111111111111111",
            "test_hash_222222222222222222222222222222",
        }

        mock_redis.smembers.return_value = token_hashes

        result = cli_runner.invoke(token, ["remove", "test_hash"])

        assert result.exit_code == 0
        assert "Multiple tokens match" in result.output
        assert "test_has...111111" in result.output
        assert "test_has...222222" in result.output

    def test_token_commands_help(self, cli_runner):
        """Test token commands help text."""
        result = cli_runner.invoke(token, ["--help"])

        assert result.exit_code == 0
        assert "Manage authentication tokens." in result.output

        # Test individual command help
        for cmd in ["add", "list", "show", "remove"]:
            result = cli_runner.invoke(token, [cmd, "--help"])
            assert result.exit_code == 0
