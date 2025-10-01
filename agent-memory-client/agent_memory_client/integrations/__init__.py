"""
Integrations for agent-memory-client with popular LLM frameworks.

This package provides seamless integration with frameworks like LangChain,
eliminating the need for manual tool wrapping.
"""

from .langchain import get_memory_tools, get_memory_tools_langchain

__all__ = ["get_memory_tools", "get_memory_tools_langchain"]
