from typing import Dict, List, Optional, Union, Any
import numpy as np
from pydantic import BaseModel, Field
from openai import AsyncOpenAI, ChatCompletion
import os
import logging
from enum import Enum
import asyncio
from redis.asyncio import ConnectionPool, Redis

# Setup logging
logger = logging.getLogger(__name__)


class MemoryMessage(BaseModel):
    """A message in the memory system"""

    role: str
    content: str


class MemoryMessagesAndContext(BaseModel):
    """Request payload for adding messages to memory"""

    messages: List[MemoryMessage]
    context: Optional[str] = None


class MemoryResponse(BaseModel):
    """Response containing messages and context"""

    messages: List[MemoryMessage]
    context: Optional[str] = None
    tokens: Optional[int] = None


class SearchPayload(BaseModel):
    """Payload for semantic search"""

    text: str


class HealthCheckResponse(BaseModel):
    """Response for health check endpoint"""

    now: int


class AckResponse(BaseModel):
    """Generic acknowledgement response"""

    status: str


class RedisearchResult(BaseModel):
    """Result from a redisearch query"""

    role: str
    content: str
    dist: float
    
    
class SearchResults(BaseModel):
    """Results from a redisearch query"""

    docs: List[RedisearchResult]
    total: int


class NamespaceQuery(BaseModel):
    """Query parameters for namespace"""

    namespace: Optional[str] = None


class GetSessionsQuery(BaseModel):
    """Query parameters for getting sessions"""

    page: int = Field(default=1)
    size: int = Field(default=20)
    namespace: Optional[str] = None


class OpenAIClientType(str, Enum):
    """Type of OpenAI client"""

    OPENAI = "openai"


class OpenAIClientWrapper:
    """Wrapper for OpenAI client"""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        """Initialize the OpenAI client based on environment variables"""

        # Regular OpenAI setup
        openai_api_base = base_url or os.environ.get("OPENAI_API_BASE")
        openai_api_key = api_key or os.environ.get("OPENAI_API_KEY")

        self.client_type = OpenAIClientType.OPENAI
        if openai_api_base:
            self.completion_client = AsyncOpenAI(
                api_key=openai_api_key, base_url=openai_api_base
            )
            self.embedding_client = AsyncOpenAI(
                api_key=openai_api_key, base_url=openai_api_base
            )
        else:
            self.completion_client = AsyncOpenAI(api_key=openai_api_key)
            self.embedding_client = AsyncOpenAI(api_key=openai_api_key)

    async def create_chat_completion(self, model: str, progressive_prompt: str) -> ChatCompletion:
        """Create a chat completion using the OpenAI API"""
        try:
            response = await self.completion_client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": progressive_prompt}]
            )
            return response
        except Exception as e:
            logger.error(f"Error creating chat completion: {e}")
            raise

    async def create_embedding(self, query_vec: List[str]) -> np.ndarray:
        """Create embeddings for the given texts"""
        try:
            embeddings = []
            embedding_model = "text-embedding-ada-002"

            # Process in batches of 20 to avoid rate limits
            batch_size = 20
            for i in range(0, len(query_vec), batch_size):
                batch = query_vec[i : i + batch_size]
                response = await self.embedding_client.embeddings.create(
                    model=embedding_model, input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)

            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise
