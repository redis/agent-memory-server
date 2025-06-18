import os
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic_settings import BaseSettings


load_dotenv()


def load_yaml_settings():
    config_path = os.getenv("APP_CONFIG_FILE", "config.yaml")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    long_term_memory: bool = True
    window_size: int = 20
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    generation_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    port: int = 8000
    mcp_port: int = 9000

    # Long-term memory backend configuration
    long_term_memory_backend: str = (
        "redis"  # redis, chroma, pinecone, weaviate, qdrant, etc.
    )

    # Redis backend settings (existing)
    # redis_url already defined above

    # Chroma backend settings
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection_name: str = "agent_memory"
    chroma_persist_directory: str | None = None

    # Pinecone backend settings
    pinecone_api_key: str | None = None
    pinecone_environment: str | None = None
    pinecone_index_name: str = "agent-memory"

    # Weaviate backend settings
    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: str | None = None
    weaviate_class_name: str = "AgentMemory"

    # Qdrant backend settings
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "agent_memory"

    # Milvus backend settings
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_name: str = "agent_memory"
    milvus_user: str | None = None
    milvus_password: str | None = None

    # PostgreSQL/PGVector backend settings
    postgres_url: str | None = None
    postgres_table_name: str = "agent_memory"

    # LanceDB backend settings
    lancedb_uri: str = "./lancedb"
    lancedb_table_name: str = "agent_memory"

    # OpenSearch backend settings
    opensearch_url: str = "http://localhost:9200"
    opensearch_username: str | None = None
    opensearch_password: str | None = None
    opensearch_index_name: str = "agent-memory"

    # The server indexes messages in long-term memory by default. If this
    # setting is enabled, we also extract discrete memories from message text
    # and save them as separate long-term memory records.
    enable_discrete_memory_extraction: bool = True

    # Topic modeling
    topic_model_source: Literal["BERTopic", "LLM"] = "LLM"
    topic_model: str = (
        "MaartenGr/BERTopic_Wikipedia"  # Use an LLM model name here if using LLM
    )
    enable_topic_extraction: bool = True
    top_k_topics: int = 3

    # Used for extracting entities from text
    ner_model: str = "dbmdz/bert-large-cased-finetuned-conll03-english"
    enable_ner: bool = True

    # RedisVL Settings (kept for backwards compatibility)
    redisvl_distance_metric: str = "COSINE"
    redisvl_vector_dimensions: str = "1536"
    redisvl_index_name: str = "memory_idx"
    redisvl_index_prefix: str = "memory_idx"

    # Docket settings
    docket_name: str = "memory-server"
    use_docket: bool = True

    # OAuth2/JWT Authentication settings
    disable_auth: bool = False
    oauth2_issuer_url: str | None = None
    oauth2_audience: str | None = None
    oauth2_jwks_url: str | None = None
    oauth2_algorithms: list[str] = ["RS256"]

    # Auth0 Client Credentials (for testing and client applications)
    auth0_client_id: str | None = None
    auth0_client_secret: str | None = None

    # Other Application settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Load YAML config first, then let env vars override
yaml_settings = load_yaml_settings()
settings = Settings(**yaml_settings)
