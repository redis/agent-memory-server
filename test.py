import os

from agent_memory_client import create_memory_client
from agent_memory_client.integrations.langchain import get_memory_tools
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


load_dotenv()


async def main():
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable is required")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        return
    # Initialize memory client
    memory_client = await create_memory_client("http://localhost:8000")

    # Get LangChain-compatible tools automatically
    tools = get_memory_tools(
        memory_client=memory_client, session_id="my_session", user_id="alice"
    )

    # Create the prompt with proper format for tool-calling agents
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant with memory capabilities."),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    # Use with LangChain agents - no manual @tool wrapping needed!
    llm = ChatOpenAI(model="gpt-4o")
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    result = await executor.ainvoke(
        {"input": "What is the capital of France? Remember that I like France."}
    )
    print("\n" + "=" * 60)
    print("RESULT:", result["output"])
    print("=" * 60)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
