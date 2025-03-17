from fastapi import APIRouter, HTTPException, Request
from typing import List
import logging

from models import SearchPayload, SearchResults
from long_term_memory import search_messages
from utils import get_redis_conn, get_openai_client
from config import settings
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/sessions/{session_id}/retrieval", response_model=SearchResults)
async def run_retrieval(
    session_id: str,
    payload: SearchPayload,
):
    """
    Run a semantic search on the messages in the session

    Args:
        session_id: The session ID
        payload: Search payload with text to search for

    Returns:
        List of search results
    """
    redis = get_redis_conn()

    if not settings.long_term_memory:
        raise HTTPException(status_code=400, detail="Long term memory is disabled")

    # For embeddings, we always use OpenAI models since Anthropic doesn't support embeddings
    client = await get_openai_client()

    try:
        results = await search_messages(payload.text, session_id, client, redis)
        return results
    except Exception as e:
        logger.error(f"Error in retrieval API: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
