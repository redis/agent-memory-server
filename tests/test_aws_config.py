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

        credentials = test_settings.aws_credentials

        assert credentials["aws_access_key_id"] is None
        assert credentials["aws_secret_access_key"] is None
        assert credentials["aws_session_token"] is None

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

    def test_titan_embed_v2_config(self):
        """Test amazon.titan-embed-text-v2:0 model config."""
        from agent_memory_server.config import MODEL_CONFIGS, ModelProvider

        config = MODEL_CONFIGS.get("amazon.titan-embed-text-v2:0")

        assert config is not None
        assert config.provider == ModelProvider.AWS_BEDROCK
        assert config.name == "amazon.titan-embed-text-v2:0"
        assert config.max_tokens == 8192
        assert config.embedding_dimensions == 1024

    def test_titan_embed_v1_config(self):
        """Test amazon.titan-embed-text-v1 model config."""
        from agent_memory_server.config import MODEL_CONFIGS, ModelProvider

        config = MODEL_CONFIGS.get("amazon.titan-embed-text-v1")

        assert config is not None
        assert config.provider == ModelProvider.AWS_BEDROCK
        assert config.name == "amazon.titan-embed-text-v1"
        assert config.max_tokens == 8192
        assert config.embedding_dimensions == 1536

    def test_cohere_embed_english_v3_config(self):
        """Test cohere.embed-english-v3 model config."""
        from agent_memory_server.config import MODEL_CONFIGS, ModelProvider

        config = MODEL_CONFIGS.get("cohere.embed-english-v3")

        assert config is not None
        assert config.provider == ModelProvider.AWS_BEDROCK
        assert config.name == "cohere.embed-english-v3"
        assert config.max_tokens == 8192
        assert config.embedding_dimensions == 1024

    def test_cohere_embed_multilingual_v3_config(self):
        """Test cohere.embed-multilingual-v3 model config."""
        from agent_memory_server.config import MODEL_CONFIGS, ModelProvider

        config = MODEL_CONFIGS.get("cohere.embed-multilingual-v3")

        assert config is not None
        assert config.provider == ModelProvider.AWS_BEDROCK
        assert config.name == "cohere.embed-multilingual-v3"
        assert config.max_tokens == 8192
        assert config.embedding_dimensions == 1024

    def test_embedding_model_config_property_with_aws_model(self):
        """Test embedding_model_config property returns correct config for AWS models."""
        from agent_memory_server.config import ModelProvider, Settings

        test_settings = Settings(embedding_model="amazon.titan-embed-text-v2:0")

        config = test_settings.embedding_model_config

        assert config is not None
        assert config.provider == ModelProvider.AWS_BEDROCK
        assert config.embedding_dimensions == 1024
