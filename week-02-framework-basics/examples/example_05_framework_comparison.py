"""
Example 5: Framework Comparison — LangGraph vs ADK Side-by-Side
=================================================================
The same task implemented in both frameworks, so you can directly
compare the developer experience, code structure, and trade-offs.

Task: "Calculate 25 * 4, then reverse the result as a string"
This requires two tool calls — perfect for comparing how each
framework handles the tool-calling loop.

Run: python week-02-framework-basics/examples/example_05_framework_comparison.py
"""

import asyncio
import logging
import os
import time
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

logging.getLogger("google_genai.types").setLevel(logging.ERROR)


# ══════════════════════════════════════════════════════════════
# SHARED TOOLS — Same logic, different wrappers per framework
# ══════════════════════════════════════════════════════════════

def _calculate_logic(expression: str) -> str:
    """Core calculation logic shared by both frameworks."""
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters"
        return f"{expression} = {eval(expression)}"
    except Exception as e:
        return f"Error: {e}"


def _reverse_logic(text: str) -> str:
    """Core string reversal logic shared by both frameworks."""
    return text[::-1]


# ══════════════════════════════════════════════════════════════
# LANGGRAPH VERSION
# ══════════════════════════════════════════════════════════════

def run_langgraph(query: str) -> dict:
    """Run a query using LangGraph. Returns result and timing."""
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    from typing import TypedDict, Annotated
    from langgraph.graph import add_messages

    # -- LangGraph tools need the @tool decorator -----------
    @tool
    def calculate(expression: str) -> str:
        """Evaluate a math expression. Example: '25 * 4', '2 ** 10'"""
        return _calculate_logic(expression)

    @tool
    def string_reverse(text: str) -> str:
        """Reverse a string."""
        return _reverse_logic(text)

    tools = [calculate, string_reverse]

    # -- LLM setup ------------------------------------------
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    else:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    llm_with_tools = llm.bind_tools(tools)

    # -- State definition ----------------------------------
    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]

    # -- Node functions ------------------------------------
    def agent_node(state):
        for attempt in range(3):
            try:
                response = llm_with_tools.invoke(state["messages"])
                return {"messages": [response]}
            except Exception as e:
                if attempt < 2:
                    continue
                raise

    def should_continue(state):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"

    # -- Build graph ---------------------------------------
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    app = graph.compile()

    # -- Execute -------------------------------------------
    start = time.time()
    result = app.invoke({"messages": [HumanMessage(content=query)]})
    elapsed = time.time() - start

    return {
        "answer": result["messages"][-1].content,
        "time_seconds": elapsed,
        "framework": "LangGraph",
        "provider": provider,
    }


# ══════════════════════════════════════════════════════════════
# ADK VERSION
# ══════════════════════════════════════════════════════════════

async def run_adk(query: str) -> dict:
    """Run a query using ADK. Returns result and timing."""
    from google.adk.agents import LlmAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    # -- ADK tools are plain functions — no decorator ------
    def calculate(expression: str) -> str:
        """Evaluate a math expression. Example: '25 * 4', '2 ** 10'"""
        return _calculate_logic(expression)

    def string_reverse(text: str) -> str:
        """Reverse a string."""
        return _reverse_logic(text)

    # -- Create agent (one-step configuration) -------------
    agent = LlmAgent(
        name="comparison_agent",
        model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
        instruction="You have access to calculate and string_reverse tools. "
                    "Use them to answer the user's question. Show the results.",
        tools=[calculate, string_reverse],
    )

    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="compare", session_service=session_service)
    session = await session_service.create_session(app_name="compare", user_id="user1")

    # -- Execute -------------------------------------------
    start = time.time()
    answer = ""
    async for event in runner.run_async(
        user_id="user1",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=query)]),
    ):
        if event.is_final_response():
            answer = event.content.parts[0].text
    elapsed = time.time() - start

    return {
        "answer": answer,
        "time_seconds": elapsed,
        "framework": "ADK",
        "provider": "Google Gemini",
    }


# ══════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ══════════════════════════════════════════════════════════════

def print_comparison(lg_result: dict, adk_result: dict):
    """Print a formatted comparison table."""

    print("\n" + "=" * 70)
    print("FRAMEWORK COMPARISON TABLE")
    print("=" * 70)

    rows = [
        ("Aspect", "LangGraph", "ADK"),
        ("-" * 25, "-" * 20, "-" * 20),
        ("Tool definition", "@tool decorator", "Plain functions"),
        ("Tool binding", "llm.bind_tools()", "tools=[...] in agent"),
        ("Graph building", "Manual (StateGraph)", "Automatic (hidden)"),
        ("State management", "TypedDict + reducers", "Session (automatic)"),
        ("Routing logic", "add_conditional_edges", "Built-in loop"),
        ("LLM providers", "Any LangChain LLM", "Gemini only"),
        ("Async required", "No (sync by default)", "Yes (async only)"),
        ("Lines of code*", "~35 lines", "~15 lines"),
        ("Control level", "Full (you own the loop)", "Limited (ADK owns it)"),
        ("Provider", lg_result["provider"], adk_result["provider"]),
        ("Execution time", f"{lg_result['time_seconds']:.2f}s", f"{adk_result['time_seconds']:.2f}s"),
    ]

    for row in rows:
        print(f"  {row[0]:<25} {row[1]:<20} {row[2]:<20}")

    print("\n  * Approximate lines for the agent setup (excluding tool definitions)")

    print(f"\n{'-' * 70}")
    print(f"LangGraph answer: {lg_result['answer'][:100]}...")
    print(f"ADK answer:       {adk_result['answer'][:100]}...")
    print(f"{'-' * 70}")

    print("""
WHEN TO USE WHICH:

  LangGraph is better when you need:
    + Full control over the execution flow
    + Custom routing logic (e.g., multi-agent handoffs)
    + Multiple LLM providers (Groq, OpenAI, Anthropic...)
    + Complex state with custom reducers
    + Checkpointing and resumable workflows

  ADK is better when you need:
    + Rapid prototyping -- get an agent running fast
    + Simple tool integration -- no boilerplate
    + Built-in session management
    + Google Cloud deployment (Cloud Run, Vertex AI)
    + Multi-agent systems with minimal wiring (Week 4+)
""")


# ══════════════════════════════════════════════════════════════
# Run the comparison
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\nFramework Comparison: LangGraph vs ADK")
    print("=" * 60)

    query = "Calculate 25 * 4, then reverse the result as a string"
    print(f"Task: {query}\n")

    # Run LangGraph version
    print("Running LangGraph...")
    try:
        lg_result = run_langgraph(query)
        print(f"  [x] Done in {lg_result['time_seconds']:.2f}s")
    except Exception as e:
        print(f"  [ ] Error: {e}")
        lg_result = {"answer": f"Error: {e}", "time_seconds": 0, "framework": "LangGraph", "provider": "N/A"}

    # Run ADK version
    print("Running ADK...")
    try:
        adk_result = asyncio.run(run_adk(query))
        print(f"  [x] Done in {adk_result['time_seconds']:.2f}s")
    except Exception as e:
        print(f"  [ ] Error: {e}")
        adk_result = {"answer": f"Error: {e}", "time_seconds": 0, "framework": "ADK", "provider": "N/A"}

    # Print comparison
    print_comparison(lg_result, adk_result)
