import logging
import os
from enum import Enum
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings


load_dotenv()


logger = logging.getLogger(__name__)


class ModelProvider(str, Enum):
    """Type of model provider"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AWS_BEDROCK = "aws-bedrock"


class ModelConfig(BaseModel):
    """Configuration for a model"""

    provider: ModelProvider
    name: str
    max_tokens: int
    embedding_dimensions: int = 1536  # Default for OpenAI ada-002


# Model configuration mapping
MODEL_CONFIGS = {
    # OpenAI Models
    "gpt-3.5-turbo": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="gpt-3.5-turbo",
        max_tokens=4096,
        embedding_dimensions=1536,
    ),
    "gpt-3.5-turbo-16k": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="gpt-3.5-turbo-16k",
        max_tokens=16384,
        embedding_dimensions=1536,
    ),
    "gpt-4": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="gpt-4",
        max_tokens=8192,
        embedding_dimensions=1536,
    ),
    "gpt-4-32k": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="gpt-4-32k",
        max_tokens=32768,
        embedding_dimensions=1536,
    ),
    "gpt-4o": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="gpt-4o",
        max_tokens=128000,
        embedding_dimensions=1536,
    ),
    "gpt-4o-mini": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="gpt-4o-mini",
        max_tokens=128000,
        embedding_dimensions=1536,
    ),
    # Newer reasoning models
    "o1": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="o1",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    "o1-mini": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="o1-mini",
        max_tokens=128000,
        embedding_dimensions=1536,
    ),
    "o3-mini": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="o3-mini",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    # Embedding models
    "text-embedding-ada-002": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="text-embedding-ada-002",
        max_tokens=8191,
        embedding_dimensions=1536,
    ),
    "text-embedding-3-small": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="text-embedding-3-small",
        max_tokens=8191,
        embedding_dimensions=1536,
    ),
    "text-embedding-3-large": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="text-embedding-3-large",
        max_tokens=8191,
        embedding_dimensions=3072,
    ),
    # Anthropic Models
    "claude-3-opus-20240229": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        name="claude-3-opus-20240229",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    "claude-3-sonnet-20240229": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        name="claude-3-sonnet-20240229",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    "claude-3-haiku-20240307": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        name="claude-3-haiku-20240307",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    "claude-3-5-sonnet-20240620": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        name="claude-3-5-sonnet-20240620",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    # Latest Anthropic Models
    "claude-3-7-sonnet-20250219": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        name="claude-3-7-sonnet-20250219",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    "claude-3-5-sonnet-20241022": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        name="claude-3-5-sonnet-20241022",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    "claude-3-5-haiku-20241022": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        name="claude-3-5-haiku-20241022",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    # Convenience aliases
    "claude-3-7-sonnet-latest": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        name="claude-3-7-sonnet-20250219",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    "claude-3-5-sonnet-latest": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        name="claude-3-5-sonnet-20241022",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    "claude-3-5-haiku-latest": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        name="claude-3-5-haiku-20241022",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    "claude-3-opus-latest": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        name="claude-3-opus-20240229",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    # AWS Bedrock Embedding Models
    "amazon.titan-embed-text-v2:0": ModelConfig(
        provider=ModelProvider.AWS_BEDROCK,
        name="amazon.titan-embed-text-v2:0",
        max_tokens=8192,
        embedding_dimensions=1024,
    ),
    "amazon.titan-embed-text-v1": ModelConfig(
        provider=ModelProvider.AWS_BEDROCK,
        name="amazon.titan-embed-text-v1",
        max_tokens=8192,
        embedding_dimensions=1536,
    ),
    "cohere.embed-english-v3": ModelConfig(
        provider=ModelProvider.AWS_BEDROCK,
        name="cohere.embed-english-v3",
        max_tokens=8192,
        embedding_dimensions=1024,
    ),
    "cohere.embed-multilingual-v3": ModelConfig(
        provider=ModelProvider.AWS_BEDROCK,
        name="cohere.embed-multilingual-v3",
        max_tokens=8192,
        embedding_dimensions=1024,
    ),
    # AWS Bedrock LLM Models
    "anthropic.claude-sonnet-4-5-20250929-v1:0": ModelConfig(
        provider=ModelProvider.AWS_BEDROCK,
        name="anthropic.claude-sonnet-4-5-20250929-v1:0",
        max_tokens=200000,
    ),
    "anthropic.claude-haiku-4-5-20251001-v1:0": ModelConfig(
        provider=ModelProvider.AWS_BEDROCK,
        name="anthropic.claude-haiku-4-5-20251001-v1:0",
        max_tokens=200000,
    ),
    "anthropic.claude-opus-4-5-20251101-v1:0": ModelConfig(
        provider=ModelProvider.AWS_BEDROCK,
        name="anthropic.claude-opus-4-5-20251101-v1:0",
        max_tokens=200000,
    ),
}


class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    long_term_memory: bool = True
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    openai_api_base: str | None = None
    anthropic_api_base: str | None = None
    generation_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"

    # Cloud
    ## Cloud region
    region_name: str | None = None

    ## AWS Cloud credentials
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None

    # Model selection for query optimization
    slow_model: str = "gpt-4o"  # Slower, more capable model for complex tasks
    fast_model: str = (
        "gpt-4o-mini"  # Faster, smaller model for quick tasks like query optimization
    )
    port: int = 8000
    mcp_port: int = 9000

    # Vector store factory configuration
    # Python dotted path to function that returns VectorStore or VectorStoreAdapter
    # Function signature: (embeddings: Embeddings) -> Union[VectorStore, VectorStoreAdapter]
    # Examples:
    #   - "agent_memory_server.vectorstore_factory.create_redis_vectorstore"
    #   - "my_module.my_vectorstore_factory"
    #   - "my_package.adapters.create_custom_adapter"
    vectorstore_factory: str = (
        "agent_memory_server.vectorstore_factory.create_redis_vectorstore"
    )

    # RedisVL configuration (used by default Redis factory)
    redisvl_index_name: str = "memory_records"

    # The server indexes messages in long-term memory by default. If this
    # setting is enabled, we also extract discrete memories from message text
    # and save them as separate long-term memory records.
    enable_discrete_memory_extraction: bool = True

    # Topic modeling
    topic_model_source: Literal["BERTopic", "LLM"] = "LLM"
    # If using BERTopic, use a supported model, such as
    # "MaartenGr/BERTopic_Wikipedia"
    topic_model: str = "gpt-4o-mini"
    enable_topic_extraction: bool = True
    top_k_topics: int = 3

    # Used for extracting entities from text
    ner_model: str = "dbmdz/bert-large-cased-finetuned-conll03-english"
    enable_ner: bool = True
    index_all_messages_in_long_term_memory: bool = False

    # RedisVL Settings
    # TODO: Adapt to vector store settings
    redisvl_distance_metric: str = "COSINE"
    redisvl_vector_dimensions: str = "1536"
    redisvl_index_prefix: str = "memory_idx"
    redisvl_indexing_algorithm: str = "HNSW"

    # Docket settings
    docket_name: str = "memory-server"
    use_docket: bool = True

    # Authentication settings
    disable_auth: bool = True
    auth_mode: Literal["disabled", "token", "oauth2"] = "disabled"

    # OAuth2/JWT Authentication settings
    oauth2_issuer_url: str | None = None
    oauth2_audience: str | None = None
    oauth2_jwks_url: str | None = None
    oauth2_algorithms: list[str] = ["RS256"]

    # Token Authentication settings
    token_auth_enabled: bool = False

    # Auth0 Client Credentials (for testing and client applications)
    auth0_client_id: str | None = None
    auth0_client_secret: str | None = None

    # Working memory settings
    summarization_threshold: float = (
        0.7  # Fraction of context window that triggers summarization
    )

    # Message timestamp validation settings
    # If true, reject messages without created_at timestamp.
    # If false (default), auto-generate timestamp with deprecation warning.
    require_message_timestamps: bool = False
    # Maximum allowed clock skew for future timestamp validation (in seconds)
    max_future_timestamp_seconds: int = 300  # 5 minutes

    # Working memory migration settings
    # Set to True to skip backward compatibility checks for old string-format keys.
    # Use this after running 'agent-memory migrate-working-memory' or for fresh installs.
    # When True, the server assumes all working memory keys are in JSON format,
    # skipping the startup scan and per-read type checks for better performance.
    working_memory_migration_complete: bool = False

    # Query optimization settings
    query_optimization_prompt_template: str = """Transform this natural language query into an optimized version for semantic search. The goal is to make it more effective for finding semantically similar content while preserving the original intent.

Guidelines:
- Keep the core meaning and intent
- Use more specific and descriptive terms
- Remove unnecessary words like "tell me", "I want to know", "can you"
- Focus on the key concepts and topics
- Make it concise but comprehensive

Original query: {query}

Optimized query:"""
    min_optimized_query_length: int = 2

    # Other Application settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    default_mcp_user_id: str | None = None
    default_mcp_namespace: str | None = None

    # Forgetting settings
    forgetting_enabled: bool = False
    forgetting_every_minutes: int = 60
    forgetting_max_age_days: float | None = None
    forgetting_max_inactive_days: float | None = None
    # Keep only top N most recent (by recency score) when budget is set
    forgetting_budget_keep_top_n: int | None = None

    # Compaction settings
    compaction_every_minutes: int = 10

    # Progressive summarization prompt template
    progressive_summarization_prompt: str = """You are a precise summarization assistant. Your task is to progressively
summarize conversation history while maintaining critical context and accuracy.

INSTRUCTIONS:
1. Build upon the previous summary by incorporating new information chronologically
2. Preserve key details: names, technical terms, code references, and important decisions
3. Maintain the temporal sequence of events and discussions
4. For technical discussions, keep specific terms, versions, and implementation details
5. For code-related content, preserve function names, file paths, and important parameters
6. If the new content is irrelevant or doesn't add value, return "NONE"
7. Keep the summary concise but complete - aim for 2-3 sentences unless more detail is crucial
8. Use neutral, factual language

EXAMPLE
Current summary:
The user inquires about retirement investment options, specifically comparing
traditional IRAs and Roth IRAs. The assistant explains the key differences in
tax treatment, with traditional IRAs offering immediate tax deductions and Roth
IRAs providing tax-free withdrawals in retirement.

New lines of conversation:
Human: What factors should I consider when deciding between the two?
Assistant: Several key factors influence this decision: 1) Your current tax
bracket vs. expected retirement tax bracket, 2) Time horizon until retirement,
3) Current income and eligibility for Roth IRA contributions, and 4) Desire for
flexibility in retirement withdrawals. For example, if you expect to be in a
higher tax bracket during retirement, a Roth IRA might be more advantageous
since qualified withdrawals are tax-free. Additionally, Roth IRAs don't have
required minimum distributions (RMDs) during your lifetime, offering more
flexibility in estate planning.

New summary:
The discussion covers retirement investment options, comparing traditional and
Roth IRAs' tax implications, with traditional IRAs offering immediate deductions
and Roth IRAs providing tax-free withdrawals. The conversation expands to cover
decision factors including current vs. future tax brackets, retirement timeline,
income eligibility, and withdrawal flexibility, with specific emphasis on Roth
IRA advantages for those expecting higher retirement tax brackets and the
benefit of no required minimum distributions. END OF EXAMPLE

Current summary:
{prev_summary}

New lines of conversation:
{messages_joined}

New summary:
"""

    @field_validator("progressive_summarization_prompt")
    @classmethod
    def validate_progressive_summarization_prompt(cls, v: str) -> str:
        """Validate that the progressive summarization prompt contains required placeholders."""
        required_vars = ["prev_summary", "messages_joined"]
        missing_vars = [var for var in required_vars if f"{{{var}}}" not in v]

        if missing_vars:
            raise ValueError(
                f"progressive_summarization_prompt must contain the following placeholders: "
                f"{', '.join(f'{{{var}}}' for var in missing_vars)}"
            )

        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra environment variables

    @property
    def generation_model_config(self) -> ModelConfig | None:
        """Get configuration for the generation model."""
        return MODEL_CONFIGS.get(self.generation_model)

    @property
    def embedding_model_config(self) -> ModelConfig | None:
        """Get configuration for the embedding model."""
        return MODEL_CONFIGS.get(self.embedding_model)

    @property
    def aws_credentials(self) -> dict[str, str]:
        """
        Get a dictionary of AWS credentials.
        """
        if not self.aws_access_key_id or not self.aws_secret_access_key:
            err_msg = "AWS access key ID and secret access key are missing. Please set them in the environment."
            logger.error(err_msg)
            raise ValueError(err_msg)

        credentials: dict[str, str] = {
            "aws_access_key_id": self.aws_access_key_id,
            "aws_secret_access_key": self.aws_secret_access_key,
        }
        if self.aws_session_token:
            # The session token is optional (only for STS).
            credentials["aws_session_token"] = self.aws_session_token

        return credentials

    @property
    def aws_region(self) -> str:
        """
        Get the AWS region.
        """
        if not self.region_name:
            raise ValueError("Missing environment variable 'REGION_NAME'.")
        return self.region_name

    def load_yaml_config(self, config_path: str) -> dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            return {}
        with open(config_path) as f:
            return yaml.safe_load(f) or {}


settings = Settings()


def get_config():
    """Get configuration from environment and settings files."""
    config_data = {}

    # If REDIS_MEMORY_CONFIG is set, load config from file
    config_file = os.getenv("REDIS_MEMORY_CONFIG")
    if config_file:
        try:
            with open(config_file) as f:
                if config_file.endswith((".yaml", ".yml")):
                    config_data = yaml.safe_load(f) or {}
                else:
                    # Assume JSON
                    import json

                    config_data = json.load(f) or {}
        except FileNotFoundError:
            print(f"Warning: Config file {config_file} not found")
        except Exception as e:
            print(f"Warning: Error loading config file {config_file}: {e}")

    # Environment variables override file config
    for key, value in os.environ.items():
        if key.startswith("REDIS_MEMORY_"):
            config_key = key[13:].lower()  # Remove REDIS_MEMORY_ prefix
            config_data[config_key] = value

    return config_data
