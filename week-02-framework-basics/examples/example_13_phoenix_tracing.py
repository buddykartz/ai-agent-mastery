"""
Example 13: Tracing Agent Flows in Phoenix
=============================================
Phoenix (Arize) gives you a visual dashboard to inspect:
  - Every LLM call (prompt, response, tokens, latency)
  - Tool calls as child spans
  - The full agent loop as a trace
  - Token costs and error rates

Week 1 introduced Phoenix basics. This example shows how to trace
Week 2's more complex multi-tool agent flows and interpret the results.

Setup: pip install arize-phoenix openinference-instrumentation-langchain

Run: python week-02-framework-basics/examples/example_13_phoenix_tracing.py
"""

import os
import sys
import warnings
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

# Suppress SQLAlchemy reflection warnings from Phoenix
warnings.filterwarnings("ignore", message=".*Skipped unsupported reflection.*")

# Fix Windows console encoding for Phoenix emoji output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ================================================================
# Step 1: Launch Phoenix and Instrument LangChain
# ================================================================

def setup_phoenix():
    """Start Phoenix dashboard and enable LangChain tracing."""
    try:
        import phoenix as px
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from phoenix.otel import register

        px.launch_app()
        tracer_provider = register()
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

        print("[OK] Phoenix tracing is active!")
        print("Dashboard: http://localhost:6006\n")
        return True
    except ImportError:
        print("[SKIP] Phoenix not installed.")
        print("Run: pip install arize-phoenix openinference-instrumentation-langchain")
        print("Continuing without tracing...\n")
        return False


# ================================================================
# Step 2: Build a Multi-Tool Agent (traced automatically)
# ================================================================

def build_traced_agent():
    """Build a LangGraph tool agent. Phoenix traces it automatically."""
    import requests
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    from typing import TypedDict, Annotated
    from langgraph.graph import add_messages

    @tool
    def get_weather(city: str) -> str:
        """Get current weather for a city."""
        try:
            geo = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1}, timeout=10,
            ).json()
            if geo.get("results"):
                loc = geo["results"][0]
                w = requests.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={"latitude": loc["latitude"],
                             "longitude": loc["longitude"],
                             "current_weather": True},
                    timeout=10,
                ).json()
                cw = w["current_weather"]
                return f"Weather in {loc['name']}: {cw['temperature']}C, wind {cw['windspeed']} km/h"
        except Exception:
            pass
        return f"Could not fetch weather for {city}"

    @tool
    def calculate(expression: str) -> str:
        """Evaluate a math expression."""
        try:
            allowed = set("0123456789+-*/.() ")
            if not all(c in allowed for c in expression):
                return "Error: Invalid characters"
            return f"{expression} = {eval(expression)}"
        except Exception as e:
            return f"Error: {e}"

    tools = [get_weather, calculate]

    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    else:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    llm_with_tools = llm.bind_tools(tools)

    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]

    def agent_node(state):
        for attempt in range(3):
            try:
                return {"messages": [llm_with_tools.invoke(state["messages"])]}
            except Exception:
                if attempt < 2:
                    continue
                raise

    def should_continue(state):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")

    return graph.compile(), provider


# ================================================================
# Step 3: Run Queries and Inspect Traces
# ================================================================

if __name__ == "__main__":
    print("\nPhoenix Tracing for Week 2 Agent Flows")
    print("=" * 60)

    tracing_active = setup_phoenix()

    app, provider = build_traced_agent()
    print(f"Agent ready (provider: {provider})")
    print("Tools: get_weather, calculate\n")

    from langchain_core.messages import HumanMessage

    # Run several queries to generate interesting traces
    queries = [
        # Simple: 1 LLM call, no tools
        "What is the capital of France?",
        # Medium: 1 tool call (2 LLM calls total)
        "What's the weather in Tokyo?",
        # Complex: 2 tool calls (3+ LLM calls)
        "What's the weather in London and calculate 25 * 17?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"Query {i}: {query}")
        try:
            result = app.invoke({"messages": [HumanMessage(content=query)]})
            print(f"Agent: {result['messages'][-1].content}\n")
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}\n")

    # Guide the user on what to look for
    print("=" * 60)
    print("WHAT TO LOOK FOR IN PHOENIX (http://localhost:6006):")
    print("=" * 60)
    print("""
  1. TRACES TAB:
     - Each query appears as a separate trace
     - Click a trace to see the full span tree

  2. SPAN TREE (inside a trace):
     - Root span: the full agent invocation
     - Child spans: individual LLM calls
     - Grandchild spans: tool executions
     - Query 1 has ~1 span (direct answer)
     - Query 2 has ~3 spans (LLM -> tool -> LLM)
     - Query 3 has ~5+ spans (multiple tool calls)

  3. SPAN DETAILS (click a span):
     - Input: the full prompt sent to the LLM
     - Output: the LLM's response
     - Metadata: model, token count, latency
     - Tool calls: function name and arguments

  4. KEY METRICS:
     - Latency: how long each call took
     - Token count: input vs output tokens
     - Error rate: which calls failed
""")

    if tracing_active:
        print("Phoenix dashboard is running. Type 'exit' to quit.\n")
        while True:
            try:
                user_input = input("> ").strip().lower()
                if user_input in ("exit", "quit"):
                    print("Shutting down...")
                    break
            except (KeyboardInterrupt, EOFError):
                print("\nShutting down...")
                break

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    else:
        print("(Install Phoenix to see the dashboard)")
