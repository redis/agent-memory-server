from unittest.mock import MagicMock, patch
from agent_memory_server._aws.clients import create_aws_session, create_bedrock_client
from agent_memory_server.config import Settings


class TestCreateAwsSession:
    """Test the create_aws_session function."""

    def test_create_aws_session_with_defaults(self):
        """Test creating an AWS session with default settings."""
        mock_settings = Settings(
            region_name="us-west-2",
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            aws_session_token="test-token",
        )

        with patch("agent_memory_server._aws.clients.settings", new=mock_settings):
            session = create_aws_session()
            credentials = session.get_credentials()

            assert session.region_name == "us-west-2"
            assert credentials.access_key == "test-key"
            assert credentials.secret_key == "test-secret"
            assert credentials.token == "test-token"

    def test_create_aws_session_with_explicit_region(self):
        """Test creating an AWS session with an explicit region."""
        mock_settings = Settings(
            region_name="eu-west-2",
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            aws_session_token="test-token",
        )

        with patch("agent_memory_server._aws.clients.settings", new=mock_settings):
            session = create_aws_session(region_name="eu-west-1")
            credentials = session.get_credentials()
            assert session.region_name == "eu-west-1"
            assert credentials.access_key == "test-key"
            assert credentials.secret_key == "test-secret"
            assert credentials.token == "test-token"

    def test_create_aws_session_with_explicit_credentials(self):
        """Test creating an AWS session with explicit credentials."""
        mock_settings = Settings(
            region_name="us-east-1",
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            aws_session_token="test-token",
        )

        with patch("agent_memory_server._aws.clients.settings", new=mock_settings):
            session = create_aws_session(
                credentials={
                    "aws_access_key_id": "test-key-2",
                    "aws_secret_access_key": "test-secret-2",
                    "aws_session_token": "test-token-2",
                }
            )
            credentials = session.get_credentials()
            assert session.region_name == "us-east-1"
            assert credentials.access_key == "test-key-2"
            assert credentials.secret_key == "test-secret-2"
            assert credentials.token == "test-token-2"

class TestCreateBedrockClient:
    """Test the create_bedrock_client function."""

    def test_create_bedrock_client_with_defaults(self):
        """Test creating a Bedrock client with default session."""
        mock_settings = Settings(
            region_name="us-east-1",
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            aws_session_token="test-token",
        )

        with patch("agent_memory_server._aws.clients.settings", new=mock_settings):
            with patch(
                "agent_memory_server._aws.clients.create_aws_session"
            ) as mock_create_session:
                mock_session = MagicMock()
                mock_client = MagicMock()
                mock_session.client.return_value = mock_client
                mock_create_session.return_value = mock_session

                result = create_bedrock_client()

                mock_create_session.assert_called_once_with(region_name=None)
                mock_session.client.assert_called_once_with(
                    "bedrock", region_name="us-east-1"
                )
                assert result == mock_client

    def test_create_bedrock_client_with_explicit_region(self):
        """Test creating Bedrock client with explicit region."""
        mock_settings = Settings(
            region_name="us-east-1",
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            aws_session_token="test-token",
        )

        with patch("agent_memory_server._aws.clients.settings", new=mock_settings):
            with patch(
                "agent_memory_server._aws.clients.create_aws_session"
            ) as mock_create_session:
                mock_session = MagicMock()
                mock_client = MagicMock()
                mock_session.client.return_value = mock_client
                mock_create_session.return_value = mock_session

                result = create_bedrock_client(region_name="eu-central-1")

                mock_create_session.assert_called_once_with(region_name="eu-central-1")
                mock_session.client.assert_called_once_with(
                    "bedrock", region_name="eu-central-1"
                )
                assert result == mock_client

    def test_create_bedrock_client_with_existing_session(self):
        """Test creating Bedrock client with an existing session."""
        mock_settings = Settings(
            region_name="us-west-2",
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            aws_session_token="test-token",
        )

        with patch("agent_memory_server._aws.clients.settings", new=mock_settings):
            mock_session = MagicMock()
            mock_client = MagicMock()
            mock_session.client.return_value = mock_client

            result = create_bedrock_client(session=mock_session)

            mock_session.client.assert_called_once_with(
                "bedrock", region_name="us-west-2"
            )
            assert result == mock_client
