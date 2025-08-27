"""
Example usage of configurable memory storage strategies.

This demonstrates how to use the new memory strategy configuration feature
to customize how memories are extracted from working memory sessions.
"""

from agent_memory_server.memory_strategies import (
    get_memory_strategy,
)
from agent_memory_server.models import (
    MemoryMessage,
    MemoryStrategyConfig,
    WorkingMemory,
)


def demonstrate_memory_strategies():
    """Demonstrate different memory extraction strategies."""

    print("=== Redis Agent Memory Server - Configurable Memory Strategies ===\n")

    # 1. Default Strategy (Discrete)
    print("1. Default Strategy - DiscreteMemoryStrategy")
    print("   Extracts discrete semantic and episodic facts from messages")

    default_working_memory = WorkingMemory(
        session_id="session-1",
        messages=[
            MemoryMessage(
                role="user", content="I love coffee and work best in the morning"
            ),
            MemoryMessage(role="assistant", content="I'll remember your preferences!"),
        ],
        memories=[],
        # long_term_memory_strategy defaults to DiscreteMemoryStrategy
    )

    print(f"   Strategy: {default_working_memory.long_term_memory_strategy.strategy}")
    print(f"   Config: {default_working_memory.long_term_memory_strategy.config}")
    print()

    # 2. Summary Strategy
    print("2. Summary Strategy - SummaryMemoryStrategy")
    print("   Creates concise summaries of entire conversations")

    summary_config = MemoryStrategyConfig(
        strategy="summary", config={"max_summary_length": 300}
    )

    summary_working_memory = WorkingMemory(
        session_id="session-2",
        messages=[
            MemoryMessage(
                role="user", content="Let's discuss the project requirements"
            ),
            MemoryMessage(
                role="assistant",
                content="Sure! What kind of project are you working on?",
            ),
            MemoryMessage(role="user", content="A web app with React and PostgreSQL"),
        ],
        memories=[],
        long_term_memory_strategy=summary_config,
    )

    print(f"   Strategy: {summary_working_memory.long_term_memory_strategy.strategy}")
    print(f"   Config: {summary_working_memory.long_term_memory_strategy.config}")
    print()

    # 3. User Preferences Strategy
    print("3. User Preferences Strategy - UserPreferencesMemoryStrategy")
    print("   Focuses on extracting user preferences, settings, and characteristics")

    preferences_config = MemoryStrategyConfig(strategy="preferences", config={})

    preferences_working_memory = WorkingMemory(
        session_id="session-3",
        messages=[
            MemoryMessage(
                role="user", content="I always prefer dark mode and email over SMS"
            ),
            MemoryMessage(
                role="assistant", content="Got it, I'll remember your preferences"
            ),
        ],
        memories=[],
        long_term_memory_strategy=preferences_config,
    )

    print(
        f"   Strategy: {preferences_working_memory.long_term_memory_strategy.strategy}"
    )
    print(f"   Config: {preferences_working_memory.long_term_memory_strategy.config}")
    print()

    # 4. Custom Strategy
    print("4. Custom Strategy - CustomMemoryStrategy")
    print("   Uses a custom prompt for specialized extraction")

    custom_config = MemoryStrategyConfig(
        strategy="custom",
        config={
            "custom_prompt": """
            Extract technical information and decisions from this conversation: {message}

            Focus on:
            - Technology choices
            - Architecture decisions
            - Implementation details

            Return JSON with memories array containing type, text, topics, entities.
            Current datetime: {current_datetime}
            """,
        },
    )

    custom_working_memory = WorkingMemory(
        session_id="session-4",
        messages=[
            MemoryMessage(
                role="user",
                content="We decided to use Redis for caching and PostgreSQL for the main database",
            ),
            MemoryMessage(
                role="assistant",
                content="Good choices! Redis will help with performance.",
            ),
        ],
        memories=[],
        long_term_memory_strategy=custom_config,
    )

    print(f"   Strategy: {custom_working_memory.long_term_memory_strategy.strategy}")
    print(
        f"   Config keys: {list(custom_working_memory.long_term_memory_strategy.config.keys())}"
    )
    print()

    # 5. Strategy-aware MCP Tool Generation
    print("5. Strategy-aware MCP Tool Generation")
    print("   Each working memory session can generate custom MCP tools")

    # Generate strategy-aware tool description
    summary_description = (
        summary_working_memory.get_create_long_term_memory_tool_description()
    )
    print("   Summary strategy tool description:")
    print("  ", summary_description.split("\n")[0])  # First line
    print("  ", summary_description.split("\n")[4])  # Strategy description line
    print()

    # Generate strategy-aware tool function
    summary_tool = summary_working_memory.create_long_term_memory_tool()
    print(f"   Generated tool name: {summary_tool.__name__}")
    print("   Tool docstring preview:", summary_tool.__doc__.split("\n")[0])
    print()

    # 6. Using the Strategy Factory
    print("6. Using the Strategy Factory")
    print("   Get strategy instances programmatically")

    # Get different strategies
    discrete_strategy = get_memory_strategy("discrete")
    summary_strategy = get_memory_strategy("summary", max_summary_length=200)
    preferences_strategy = get_memory_strategy("preferences")

    print(f"   Discrete strategy: {discrete_strategy.__class__.__name__}")
    print(
        f"   Summary strategy: {summary_strategy.__class__.__name__} (max_length: {summary_strategy.max_summary_length})"
    )
    print(f"   Preferences strategy: {preferences_strategy.__class__.__name__}")
    print()

    print("=== Usage in Client Code ===")
    print()
    print("# When creating/updating working memory via API:")
    print("working_memory_request = {")
    print('    "session_id": "my-session",')
    print('    "messages": [{"role": "user", "content": "Hello!"}],')
    print('    "long_term_memory_strategy": {')
    print('        "strategy": "summary",')
    print('        "config": {"max_summary_length": 400}')
    print("    }")
    print("}")
    print()
    print("# The working memory will now use the summary strategy")
    print("# for background extraction when messages are promoted")
    print("# to long-term memory storage.")
    print()


if __name__ == "__main__":
    demonstrate_memory_strategies()
