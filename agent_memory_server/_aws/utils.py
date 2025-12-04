"""AWS utilities for the Agent Memory Server.

This module contains utilities for working with AWS services.
"""

from botocore.exceptions import ClientError
from cachetools import TTLCache, cached

from agent_memory_server._aws.clients import create_bedrock_client
from agent_memory_server.logging import get_logger


logger = get_logger(__name__)


@cached(cache=TTLCache(maxsize=16, ttl=60 * 60))  # 1 hour
def bedrock_embedding_model_exists(
    model_id: str,
    region_name: str | None = None,
) -> bool:
    """Returns True if a Bedrock embedding model with the given model_id exists.

    Args:
        model_id (str): The ID of the Bedrock model to check.
        region_name (str | None): The AWS region to check. If not provided, it will be picked up from the environment.

    Returns:
        True if an embedding model with the given ID exists, False otherwise.
        Returns True on credential/permission errors to allow the actual embedding call to proceed.
    """
    client = create_bedrock_client(region_name=region_name)

    try:
        response = client.list_foundation_models(byOutputModality="EMBEDDING")
        for model_info in response.get("modelSummaries", []):
            output_modalities: list[str] = model_info.get("outputModalities", [])
            is_embedding: bool = (
                "EMBEDDING" in output_modalities
                or model_info.get("modelModality") == "EMBEDDING"
            )
            if not is_embedding:
                continue
            maybe_model_id: str | None = model_info.get("modelId")
            if maybe_model_id == model_id:
                return True
        return False
    except ClientError:
        logger.exception(
            f"Error checking if Bedrock embedding model {model_id} exists. "
            "Defaulting to False."
        )
        return False


@cached(cache=TTLCache(maxsize=16, ttl=60 * 60))  # 1 hour
def bedrock_llm_model_exists(
    model_id: str,
    region_name: str | None = None,
) -> bool:
    """Returns True if a Bedrock LLM (text generation) model with the given model_id exists.

    Args:
        model_id (str): The ID of the Bedrock model to check.
        region_name (str | None): The AWS region to check. If not provided, it will be picked up from the environment.

    Returns:
        True if a text generation model with the given ID exists, False otherwise.
        Returns True on credential/permission errors to allow the actual LLM call to proceed.
    """
    client = create_bedrock_client(region_name=region_name)

    try:
        response = client.list_foundation_models(byOutputModality="TEXT")
        for model_info in response.get("modelSummaries", []):
            output_modalities: list[str] = model_info.get("outputModalities", [])
            # Check if this is a text generation model (has TEXT in output modalities)
            is_text_model: bool = "TEXT" in output_modalities
            if not is_text_model:
                continue
            maybe_model_id: str | None = model_info.get("modelId")
            if maybe_model_id == model_id:
                return True
        return False
    except ClientError:
        logger.exception(
            f"Error checking if Bedrock LLM model {model_id} exists. "
            "Defaulting to False."
        )
        return False
