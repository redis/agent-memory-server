#!/usr/bin/env python3
"""
Memory Prompt Agent Example

This example demonstrates how to use the memory prompt feature of the Agent Memory Server.
Instead of manually managing conversation history, this agent:

1. Stores all messages in working memory
2. For each turn, uses the memory prompt to get relevant memories
3. Combines the memory prompt results with the system prompt
4. Sends the enriched context to the LLM

The memory prompt feature automatically:
- Retrieves relevant memories based on the current input
- Formats them into a context-aware prompt
- Provides both recent conversation history and relevant long-term memories

Environment variables:
- OPENAI_API_KEY: Required for OpenAI ChatGPT
- MEMORY_SERVER_URL: Memory server URL (default: http://localhost:8000)
"""

import asyncio
import logging
import os
import textwrap
from typing import Any

from agent_memory_client import (
    MemoryAPIClient,
    create_memory_client,
)
from langchain_openai import ChatOpenAI


# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Reduce third-party logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Environment setup
MEMORY_SERVER_URL = os.getenv("MEMORY_SERVER_URL", "http://localhost:8000")
DEFAULT_USER = "demo_user"

SYSTEM_PROMPT = textwrap.dedent("""
    You are a helpful and knowledgeable personal assistant. You have access to memories
    from previous conversations and can provide personalized responses based on what
    you know about the user.

    You are skilled at:
    - Remembering user preferences and past conversations
    - Providing contextual responses based on conversation history
    - Being helpful, friendly, and engaging
    - Asking relevant follow-up questions

    When you receive context from memories, use them naturally in your responses.
    If memories contain information about the user's preferences, interests, or past
    conversations, incorporate that knowledge to provide more personalized assistance.

    Be conversational and natural - don't explicitly mention that you're using memories
    unless it's relevant to the conversation.
""").strip()


class MemoryPromptAgent:
    """
    A conversational agent that demonstrates the memory prompt feature.

    This agent uses the memory prompt to automatically retrieve and format
    relevant memories for each conversation turn, providing rich context
    to the LLM without manual history management.
    """

    def __init__(self):
        self._memory_client: MemoryAPIClient | None = None
        self._setup_llm()

    def _get_namespace(self, user_id: str) -> str:
        """Generate consistent namespace for a user."""
        return f"memory_prompt_agent:{user_id}"

    async def get_client(self) -> MemoryAPIClient:
        """Get the memory client, initializing it if needed."""
        if not self._memory_client:
            self._memory_client = await create_memory_client(
                base_url=MEMORY_SERVER_URL,
                timeout=30.0,
                default_model_name="gpt-4o",
            )
        return self._memory_client

    def _setup_llm(self):
        """Set up the LLM instance."""
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    async def cleanup(self):
        """Clean up resources."""
        if self._memory_client:
            await self._memory_client.close()

    async def _add_message_to_working_memory(
        self, session_id: str, user_id: str, role: str, content: str
    ) -> None:
        """Add a message to working memory."""
        client = await self.get_client()
        await client.append_messages_to_working_memory(
            session_id=session_id,
            messages=[{"role": role, "content": content, "user_id": user_id}],
            user_id=user_id,
        )

    async def _get_memory_prompt(
        self, session_id: str, user_id: str, user_input: str
    ) -> list[dict[str, Any]]:
        """Get memory prompt with relevant context for the current input."""
        client = await self.get_client()

        # Use the memory prompt feature to get relevant memories
        result = await client.memory_prompt(
            session_id=session_id,
            query=user_input,
            # Optional parameters to control memory retrieval
            model_name="gpt-4o-mini",  # Controls token-based truncation
            long_term_search={"limit": 30},  # Controls long-term memory limit
            user_id=user_id,
        )

        return result["messages"]

    async def _generate_response(
        self, session_id: str, user_id: str, user_input: str
    ) -> str:
        """Generate a response using the LLM with memory-enriched context."""
        # Get memory prompt with relevant context
        memory_messages = await self._get_memory_prompt(session_id, user_id, user_input)

        # Add system prompt to the beginning
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add memory messages (these already include context and the user query)
        for msg in memory_messages:
            # Extract text content from the message structure
            content = msg.get("content", "")
            if isinstance(content, dict) and "text" in content:
                content = content["text"]
            messages.append({"role": msg["role"], "content": str(content)})

        # Generate response
        response = self.llm.invoke(messages)
        return str(response.content)

    async def process_user_input(
        self, user_input: str, session_id: str, user_id: str
    ) -> str:
        """Process user input and return assistant response."""
        try:
            # Add user message to working memory first
            await self._add_message_to_working_memory(
                session_id, user_id, "user", user_input
            )

            # Generate response using memory prompt
            response = await self._generate_response(session_id, user_id, user_input)

            # Add assistant response to working memory
            await self._add_message_to_working_memory(
                session_id, user_id, "assistant", response
            )

            return response

        except Exception as e:
            logger.exception(f"Error processing user input: {e}")
            return "I'm sorry, I encountered an error processing your request."

    async def run_async(
        self, session_id: str = "memory_prompt_session", user_id: str = DEFAULT_USER
    ):
        """Main async interaction loop for the memory prompt agent."""
        print("Welcome to the Memory Prompt Agent! (Type 'exit' to quit)")
        print("\nThis agent uses memory prompts to provide contextual responses.")
        print("Try mentioning your preferences, interests, or past conversations!")
        print(f"Session ID: {session_id}, User ID: {user_id}")
        print()

        try:
            while True:
                user_input = input("\nYou (type 'quit' to quit): ")

                if not user_input.strip():
                    continue

                if user_input.lower() in ["exit", "quit"]:
                    print("Thank you for using the Memory Prompt Agent. Goodbye!")
                    break

                # Process input and get response
                print("Thinking...")
                response = await self.process_user_input(
                    user_input, session_id, user_id
                )
                print(f"\nAssistant: {response}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
        finally:
            await self.cleanup()

    def run(
        self, session_id: str = "memory_prompt_session", user_id: str = DEFAULT_USER
    ):
        """Synchronous wrapper for the async run method."""
        asyncio.run(self.run_async(session_id, user_id))


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Memory Prompt Agent Example")
    parser.add_argument("--user-id", default=DEFAULT_USER, help="User ID")
    parser.add_argument(
        "--session-id", default="demo_memory_prompt_session", help="Session ID"
    )
    parser.add_argument(
        "--memory-server-url", default="http://localhost:8000", help="Memory server URL"
    )

    args = parser.parse_args()

    # Check for required API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is required")
        return

    # Set memory server URL from argument if provided
    if args.memory_server_url:
        os.environ["MEMORY_SERVER_URL"] = args.memory_server_url

    try:
        agent = MemoryPromptAgent()
        agent.run(session_id=args.session_id, user_id=args.user_id)
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        logger.error(f"Error running memory prompt agent: {e}")
        raise


if __name__ == "__main__":
    main()
