import pytest


class TestAWSConfigProperties:
    """Test cases for AWS-related config properties."""

    def test_aws_credentials_property_all_set(self):
        """Test aws_credentials property when all credentials are set."""
        from agent_memory_server.config import Settings

        test_settings = Settings(
            aws_access_key_id="test-key-id",
            aws_secret_access_key="test-secret-key",
            aws_session_token="test-session-token",
        )

        credentials = test_settings.aws_credentials

        assert credentials["aws_access_key_id"] == "test-key-id"
        assert credentials["aws_secret_access_key"] == "test-secret-key"
        assert credentials["aws_session_token"] == "test-session-token"

    def test_aws_credentials_property_none_set(self):
        """Test aws_credentials property when no credentials are set."""
        from agent_memory_server.config import Settings

        test_settings = Settings(
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_session_token=None,
        )

        with pytest.raises(ValueError):
            test_settings.aws_credentials

    def test_aws_region_property_set(self):
        """Test aws_region property when region is set."""
        from agent_memory_server.config import Settings

        test_settings = Settings(region_name="eu-west-1")

        assert test_settings.aws_region == "eu-west-1"

    def test_aws_region_property_not_set_raises(self):
        """Test aws_region property raises when region is not set."""
        from agent_memory_server.config import Settings

        test_settings = Settings(region_name=None)

        with pytest.raises(ValueError) as exc_info:
            _ = test_settings.aws_region

        assert "REGION_NAME" in str(exc_info.value)


class TestAWSBedrockModelConfigs:
    """Test cases for AWS Bedrock model configurations."""

    def test_embedding_model_config_property_with_aws_model(self):
        """Test embedding_model_config property returns correct config for AWS models."""
        from agent_memory_server.config import ModelProvider, Settings

        test_settings = Settings(embedding_model="amazon.titan-embed-text-v2:0")

        config = test_settings.embedding_model_config

        assert config is not None
        assert config.provider == ModelProvider.AWS_BEDROCK
        assert config.embedding_dimensions == 1024

    def test_generation_model_config_property_with_aws_model(self):
        """Test generation_model_config property returns correct config for AWS models."""
        from agent_memory_server.config import ModelProvider, Settings

        test_settings = Settings(generation_model="anthropic.claude-sonnet-4-5-20250929-v1:0")

        config = test_settings.generation_model_config

        assert config is not None
        assert config.provider == ModelProvider.AWS_BEDROCK
        assert config.max_tokens == 200000