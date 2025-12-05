"""AWS clients for the Agent Memory Server.

This module contains utilities for creating and managing AWS clients.
"""

from typing import TYPE_CHECKING

from boto3 import Session

from agent_memory_server.config import settings


if TYPE_CHECKING:
    from mypy_boto3_bedrock import BedrockClient
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient


def create_aws_session(
    region_name: str | None = None, credentials: dict[str, str] | None = None
) -> Session:
    """Create an AWS session.

    Args:
        credentials (dict[str, str | None]): The AWS credentials to use.

    Returns:
        An AWS session.
    """
    if credentials is None:
        credentials = settings.aws_credentials
    if region_name is None:
        region_name = settings.aws_region
    return Session(region_name=region_name, **credentials)


def create_bedrock_client(
    region_name: str | None = None,
    session: Session | None = None,
) -> "BedrockClient":
    """Create a Bedrock client.

    Args:
        region_name (str | None): The AWS region to use.\
            If not provided, it will be picked up from the environment.
        session (Session | None): The AWS session to use.\
            If not provided, a new session will be created based on the environment.
    """
    if session is None:
        session = create_aws_session(region_name=region_name)
    if region_name is None:
        region_name = settings.aws_region
    return session.client("bedrock", region_name=region_name)


def create_bedrock_runtime_client(
    region_name: str | None = None,
    session: Session | None = None,
) -> "BedrockRuntimeClient":
    """Create a Bedrock runtime client.

    Args:
        region_name (str | None): The AWS region to use.\
            If not provided, it will be picked up from the environment.
        session (Session | None): The AWS session to use.\
            If not provided, a new session will be created based on the environment.

    Returns:
        A Bedrock runtime client.
    """
    if session is None:
        session = create_aws_session(region_name=region_name)
    if region_name is None:
        region_name = settings.aws_region
    return session.client("bedrock-runtime", region_name=region_name)
