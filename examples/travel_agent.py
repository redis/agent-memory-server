#!/usr/bin/env python3
"""
Travel Agent using Agent Memory Server with Web Search

This script demonstrates how to manage both short-term and long-term agent memory
using the Agent Memory Server with optional web search capabilities. The agent manages:

1. Working memory via the memory server (session messages and data)
2. Long-term memory storage and retrieval via the memory server
3. Memory extraction and contextual retrieval
4. Conversation flow without LangGraph dependencies
5. Optional cached web search using Tavily API with Redis caching
6. Automatic discovery and use of all available memory client tools

Web search features:
- Cached web search results using Redis to avoid rate limiting
- Function calling integration with OpenAI for intelligent search decisions
- Automatic fallback when Tavily API key or Redis are not available

Environment variables:
- OPENAI_API_KEY: Required for OpenAI ChatGPT
- TAVILY_API_KEY: Optional for web search functionality
- MEMORY_SERVER_URL: Memory server URL (default: http://localhost:8000)
- REDIS_URL: Redis URL for caching (default: redis://localhost:6379)
"""

import asyncio
import json
import logging
import os
import textwrap

from agent_memory_client import (
    MemoryAPIClient,
    create_memory_client,
)
from agent_memory_client.models import (
    WorkingMemory,
)
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.callbacks.manager import CallbackManagerForToolRun
from langchain_openai import ChatOpenAI
from redis import Redis


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment setup
MEMORY_SERVER_URL = os.getenv("MEMORY_SERVER_URL", "http://localhost:8000")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DEFAULT_USER = "demo_user"

MAX_WEB_SEARCH_RESULTS = 3

SYSTEM_PROMPT = {
    "role": "system",
    "content": textwrap.dedent("""
                You are a helpful travel assistant. You can help with travel-related questions.
                You have access to conversation history and memory management tools to provide
                personalized responses.
                
                Available tools:
                
                1. **web_search** (if available): Search for current travel information, weather, 
                   events, or other up-to-date data when specifically needed.
                   
                2. **Memory Management Tools** (always available):
                   - **search_memory**: Look up previous conversations and stored information
                   - **get_working_memory**: Check current session context
                   - **add_memory_to_working_memory**: Store important preferences or information
                   - **update_working_memory_data**: Save session-specific data
                
                **Guidelines**:
                - Answer the user's actual question first and directly
                - When someone shares information (like "I like X"), simply acknowledge it naturally - don't immediately give advice or suggestions unless they ask
                - Search memory or web when it would be helpful for the current conversation
                - Don't assume the user is actively planning a trip unless they explicitly say so
                - Be conversational and natural - respond to what the user actually says
                - When sharing memories, simply state what you remember rather than turning it into advice
                - Only offer suggestions, recommendations, or tips if the user explicitly asks for them
                - Store preferences and important details, but don't be overly eager about it
                - If someone shares a preference, respond like a friend would - acknowledge it, maybe ask a follow-up question, but don't launch into advice
                
                Be helpful, friendly, and responsive. Mirror their conversational style - if they're just chatting, chat back. If they ask for help, then help.
                """),
}


class CachingTavilySearchResults(TavilySearchResults):
    """
    An interface to Tavily search that caches results in Redis.

    Caching the results of the web search allows us to avoid rate limiting,
    improve latency, and reduce costs.
    """

    def __init__(self, redis_client: Redis, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.redis_client = redis_client

    def _run(
        self,
        query: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[list[dict[str, str]] | str, dict]:
        """Use the tool."""
        cache_key = f"tavily_search:{query}"
        cached_result: str | None = self.redis_client.get(cache_key)  # type: ignore
        if cached_result:
            return json.loads(cached_result), {}
        result, raw_results = super()._run(query, run_manager)
        self.redis_client.set(cache_key, json.dumps(result), ex=60 * 60)
        return result, raw_results


class TravelAgent:
    """
    Travel Agent with comprehensive memory management capabilities.

    Uses the Agent Memory Server client with automatic discovery and integration
    of all available memory tools. Supports web search when configured.
    """

    def __init__(self):
        self._memory_client: MemoryAPIClient | None = None
        self._redis_client: Redis | None = None
        self._web_search_tool: CachingTavilySearchResults | None = None

        # Initialize LLMs and tools
        self._setup_llms()
        self._setup_tools()

    def _get_namespace(self, user_id: str) -> str:
        """Generate consistent namespace for a user."""
        return f"travel_agent:{user_id}"

    async def get_client(self) -> MemoryAPIClient:
        """Get the memory client, initializing it if needed."""
        if not self._memory_client:
            self._memory_client = await create_memory_client(
                base_url=MEMORY_SERVER_URL,
                timeout=30.0,
                default_model_name="gpt-4o",  # Configure model for auto-summarization
            )
        return self._memory_client

    def _setup_llms(self):
        """Set up the LLM instances."""
        # Define the web search tool function
        web_search_function = {
            "name": "web_search",
            "description": textwrap.dedent("""
              Search the web for current information about travel destinations,
              requirements, weather, events, or any other travel-related
              queries. Use this when you need up-to-date information that may
              not be in your training data.
            """),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant travel information",
                    }
                },
                "required": ["query"],
            },
        }

        # Set up available functions list
        available_functions = []

        # Web search (optional)
        if os.getenv("TAVILY_API_KEY"):
            available_functions.append(web_search_function)

        # Memory tools (always available) - get all available tool schemas from client
        memory_tool_schemas = MemoryAPIClient.get_all_memory_tool_schemas()

        # Extract function schemas from tool schemas
        for tool_schema in memory_tool_schemas:
            available_functions.append(tool_schema["function"])

        logger.info(
            f"Available memory tools: {[tool['function']['name'] for tool in memory_tool_schemas]}"
        )

        # Log all available tools for debugging
        all_tool_names = []
        if os.getenv("TAVILY_API_KEY"):
            all_tool_names.append("web_search")
        all_tool_names.extend(
            [tool["function"]["name"] for tool in memory_tool_schemas]
        )
        logger.info(f"Total available tools: {all_tool_names}")

        # Set up LLM with function calling
        if available_functions:
            self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7).bind_functions(
                available_functions
            )
        else:
            self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    def _setup_tools(self):
        """Set up tools including web search if Tavily API key is available."""
        # Set up Redis client for caching
        try:
            self._redis_client = Redis.from_url(REDIS_URL)
            self._redis_client.ping()
            logger.info("Redis client connected for caching")
        except Exception as e:
            logger.warning(f"Could not connect to Redis for caching: {e}")
            self._redis_client = None

        # Set up web search tool if Tavily API key is available
        if os.getenv("TAVILY_API_KEY") and self._redis_client:
            try:
                self._web_search_tool = CachingTavilySearchResults(
                    redis_client=self._redis_client, max_results=MAX_WEB_SEARCH_RESULTS
                )
                logger.info("Web search tool with caching enabled")
            except Exception as e:
                logger.warning(f"Could not set up web search tool: {e}")
                self._web_search_tool = None
        else:
            if not os.getenv("TAVILY_API_KEY"):
                logger.info("TAVILY_API_KEY not set, web search disabled")
            if not self._redis_client:
                logger.info("Redis not available, web search caching disabled")

    async def cleanup(self):
        """Clean up resources."""
        if self._memory_client:
            await self._memory_client.close()
            logger.info("Memory client closed")
        if self._redis_client:
            self._redis_client.close()
            logger.info("Redis client closed")

    async def _get_working_memory(self, session_id: str, user_id: str) -> WorkingMemory:
        """Get working memory for a session, creating it if it doesn't exist."""
        client = await self.get_client()
        result = await client.get_working_memory(
            session_id=session_id,
            namespace=self._get_namespace(user_id),
            window_size=15,
        )
        return WorkingMemory(**result.model_dump())

    async def _search_web(self, query: str) -> str:
        """Perform a web search if the tool is available."""
        if not self._web_search_tool:
            return "Web search is not available. Please ensure TAVILY_API_KEY is set and Redis is connected."

        try:
            results, _ = self._web_search_tool._run(query)
            if isinstance(results, str):
                return results

            # Format the results
            formatted_results = []
            for result in results:
                title = result.get("title", "No title")
                content = result.get("content", "No content")
                url = result.get("url", "No URL")
                formatted_results.append(f"**{title}**\n{content}\nSource: {url}")

            return "\n\n".join(formatted_results)
        except Exception as e:
            logger.error(f"Error performing web search: {e}")
            return f"Error performing web search: {str(e)}"

    async def _add_message_to_working_memory(
        self, session_id: str, user_id: str, role: str, content: str
    ) -> WorkingMemory:
        """Add a message to working memory."""
        # Add new message
        new_message = [{"role": role, "content": content}]

        # Get the memory client and save updated working memory
        client = await self.get_client()
        await client.append_messages_to_working_memory(
            session_id=session_id,
            messages=new_message,
            namespace=self._get_namespace(user_id),
        )

    async def _handle_function_call(
        self,
        function_call: dict,
        context_messages: list,
        session_id: str,
        user_id: str,
    ) -> str:
        """Handle function calls for both web search and memory tools."""
        function_name = function_call["name"]

        # Handle web search separately (not a memory function)
        if function_name == "web_search":
            return await self._handle_web_search_call(function_call, context_messages)

        # Handle all memory functions using the client's unified resolver
        return await self._handle_memory_tool_call(
            function_call, context_messages, session_id, user_id
        )

    async def _handle_web_search_call(
        self, function_call: dict, context_messages: list
    ) -> str:
        """Handle web search function calls."""
        print("Searching the web...")
        try:
            function_args = json.loads(function_call["arguments"])
            query = function_args.get("query", "")

            # Perform the web search
            search_results = await self._search_web(query)

            # Generate a follow-up response with the search results
            follow_up_messages = context_messages + [
                {
                    "role": "assistant",
                    "content": f"I'll search for that information: {query}",
                },
                {
                    "role": "function",
                    "name": "web_search",
                    "content": search_results,
                },
                {
                    "role": "user",
                    "content": "Please provide a helpful response based on the search results.",
                },
            ]

            final_response = self.llm.invoke(follow_up_messages)
            return str(final_response.content)

        except (json.JSONDecodeError, TypeError):
            logger.error(f"Invalid web search arguments: {function_call['arguments']}")
            return "I'm sorry, I encountered an error processing your web search request. Please try again."

    async def _handle_memory_tool_call(
        self,
        function_call: dict,
        context_messages: list,
        session_id: str,
        user_id: str,
    ) -> str:
        """Handle memory tool function calls using the client's unified resolver."""
        function_name = function_call["name"]
        client = await self.get_client()

        print("Accessing memory...")
        result = await client.resolve_tool_call(
            tool_call=function_call,  # Pass the entire function call object
            session_id=session_id,
            namespace=self._get_namespace(user_id),
        )

        if not result["success"]:
            logger.error(f"Function call failed: {result['error']}")
            return result["formatted_response"]

        # Generate a follow-up response with the function result
        follow_up_messages = context_messages + [
            {
                "role": "assistant",
                "content": f"Let me {function_name.replace('_', ' ')}...",
            },
            {
                "role": "function",
                "name": function_name,
                "content": result["formatted_response"],
            },
            {
                "role": "user",
                "content": "Please provide a helpful response based on this information.",
            },
        ]

        final_response = self.llm.invoke(follow_up_messages)
        return str(final_response.content)

    async def _generate_response(
        self, session_id: str, user_id: str, user_input: str
    ) -> str:
        """Generate a response using the LLM with conversation context."""
        # Manage conversation history
        working_memory = await self._get_working_memory(session_id, user_id)
        context_messages = working_memory.messages

        # Always ensure system prompt is at the beginning
        # Remove any existing system messages and add our current one
        context_messages = [
            msg for msg in context_messages if msg.get("role") != "system"
        ]
        context_messages.insert(0, SYSTEM_PROMPT)

        # Note: user input has already been added to working memory,
        # so we don't need to add it again here

        try:
            logger.info(f"Context messages: {context_messages}")
            response = self.llm.invoke(context_messages)

            # Handle function calls using unified approach
            if (
                hasattr(response, "additional_kwargs")
                and "function_call" in response.additional_kwargs
            ):
                return await self._handle_function_call(
                    response.additional_kwargs["function_call"],
                    context_messages,
                    session_id,
                    user_id,
                )

            return str(response.content)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error processing your request."

    async def process_user_input(
        self, user_input: str, session_id: str, user_id: str
    ) -> str:
        """Process user input and return assistant response."""
        try:
            # Add user message to working memory first
            await self._add_message_to_working_memory(
                session_id, user_id, "user", user_input
            )

            response = await self._generate_response(session_id, user_id, user_input)
            await self._add_message_to_working_memory(
                session_id, user_id, "assistant", response
            )
            return response

        except Exception as e:
            logger.exception(f"Error processing user input: {e}")
            return "I'm sorry, I encountered an error processing your request."

    async def run_async(
        self, session_id: str = "travel_session", user_id: str = DEFAULT_USER
    ):
        """Main async interaction loop for the travel agent."""
        print("Welcome to the Travel Assistant! (Type 'exit' to quit)")
        print(f"Session ID: {session_id}, User ID: {user_id}")
        print()

        try:
            while True:
                user_input = input("\nYou (type 'quit' to quit): ")

                if not user_input.strip():
                    continue

                if user_input.lower() in ["exit", "quit"]:
                    print("Thank you for using the Travel Assistant. Goodbye!")
                    break

                # Process input and get response
                response = await self.process_user_input(
                    user_input, session_id, user_id
                )
                print(f"\nAssistant: {response}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
        finally:
            await self.cleanup()

    def run(self, session_id: str = "travel_session", user_id: str = DEFAULT_USER):
        """Synchronous wrapper for the async run method."""
        asyncio.run(self.run_async(session_id, user_id))


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Travel Agent with Memory Server")
    parser.add_argument("--user-id", default=DEFAULT_USER, help="User ID")
    parser.add_argument(
        "--session-id", default="demo_travel_session", help="Session ID"
    )
    parser.add_argument(
        "--memory-server-url", default="http://localhost:8000", help="Memory server URL"
    )
    parser.add_argument(
        "--redis-url", default="redis://localhost:6379", help="Redis URL for caching"
    )

    args = parser.parse_args()

    # Check for required API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is required")
        return

    # Check for optional Tavily API key
    if not os.getenv("TAVILY_API_KEY"):
        print(
            "Note: TAVILY_API_KEY not set - web search functionality will be disabled"
        )
        print("To enable web search, set TAVILY_API_KEY environment variable")

    # Check for Redis connection
    redis_url = args.redis_url if hasattr(args, "redis_url") else REDIS_URL
    print(f"Using Redis at: {redis_url} for caching (if available)")

    # Set memory server URL from argument if provided
    if args.memory_server_url:
        os.environ["MEMORY_SERVER_URL"] = args.memory_server_url

    # Set Redis URL from argument if provided
    if args.redis_url:
        os.environ["REDIS_URL"] = args.redis_url

    try:
        agent = TravelAgent()
        agent.run(session_id=args.session_id, user_id=args.user_id)
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        logger.error(f"Error running travel agent: {e}")
        raise


if __name__ == "__main__":
    main()
