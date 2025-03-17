from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    long_term_memory: bool = True
    window_size: int = 20
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    generation_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    port: int = 8000

settings = Settings()
