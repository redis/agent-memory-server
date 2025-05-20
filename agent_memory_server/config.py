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

    # Topic and NER model settings
    topic_model_source: Literal["NER", "LLM"] = "LLM"
    topic_model: str = "MaartenGr/BERTopic_Wikipedia"  # LLM model here if using LLM
    ner_model: str = "dbmdz/bert-large-cased-finetuned-conll03-english"
    enable_topic_extraction: bool = True
    enable_ner: bool = True
    top_k_topics: int = 3

    # RedisVL Settings
    redisvl_distance_metric: str = "COSINE"
    redisvl_vector_dimensions: str = "1536"
    redisvl_index_name: str = "memory"
    redisvl_index_prefix: str = "memory"

    # Docket settings
    docket_name: str = "memory-server"
    use_docket: bool = True

    # Other Application settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Load YAML config first, then let env vars override
yaml_settings = load_yaml_settings()
settings = Settings(**yaml_settings)
