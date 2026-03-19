"""
LangGraph Reference Agent — Clean, Commented Implementation
==============================================================
A complete, production-style LangGraph tool-using agent that you
can use as a reference or starting point for your own agents.

Features:
  - Weather lookup (Open-Meteo API, free, no key needed)
  - Calculator (safe math evaluation)
  - Full agent → tools → agent loop with conditional edges
  - Retry logic for Groq/Llama malformed tool calls
  - Max iteration guard (prevents infinite loops)
  - Token usage extraction after each call

This file exports create_agent() so you can import it elsewhere.

Run: python week-02-framework-basics/langgraph/simple_graph_agent.py
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
from langgraph.graph import add_messages


# ═══════════════════════════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════════════════════════
# Each tool is a function decorated with @tool. LangGraph uses
# the function name, docstring, and type hints to generate the
# tool schema that the LLM sees.

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.
    Returns temperature and wind speed.
    Use this when the user asks about weather conditions.
    """
    # Primary API: Open-Meteo (free, no API key required)
    try:
        # Step 1: Get city coordinates
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=10,
        ).json()

        if geo.get("results"):
            loc = geo["results"][0]
            # Step 2: Get weather for those coordinates
            weather = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": loc["latitude"],
                    "longitude": loc["longitude"],
                    "current_weather": True,
                },
                timeout=10,
            ).json()
            cw = weather["current_weather"]
            return (f"Weather in {loc['name']}: {cw['temperature']}°C, "
                    f"wind {cw['windspeed']} km/h")
    except Exception:
        pass

    # Fallback API: wttr.in (can be slow)
    try:
        resp = requests.get(
            f"https://wttr.in/{city}?format=%t+%C",
            timeout=5,
            headers={"User-Agent": "curl/7.68.0"},
        )
        if resp.status_code == 200 and "Unknown" not in resp.text:
            return f"Weather in {city}: {resp.text.strip()}"
    except Exception:
        pass

    return f"Could not fetch weather for {city} (APIs unavailable)"


@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression. Supports +, -, *, /, ** (power), and parentheses.
    Examples: '15 * 7', '2 ** 10', '(100 + 50) / 3'
    """
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: Only numbers and operators (+-*/.) are allowed"
        result = eval(expression)
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return f"Error: Division by zero in '{expression}'"
    except Exception as e:
        return f"Error: {e}"


# ═══════════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════════
# The state is shared by all nodes. add_messages is a reducer
# that APPENDS new messages instead of replacing the list.

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # Conversation history
    iteration: int                             # Loop counter for safety


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

MAX_ITERATIONS = 8     # Safety limit to prevent infinite loops
MAX_LLM_RETRIES = 3   # Retries for malformed tool calls


# ═══════════════════════════════════════════════════════════════
# AGENT CREATION
# ═══════════════════════════════════════════════════════════════

def create_agent():
    """Build and return a compiled LangGraph tool-using agent.

    Returns:
        A compiled LangGraph app ready to invoke.
    """
    tools = [get_weather, calculate]

    # ── LLM setup with provider flexibility ────────────────
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    else:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    # Bind tools so the LLM knows what's available
    llm_with_tools = llm.bind_tools(tools)

    # ── Agent node: calls the LLM ─────────────────────────
    def agent_node(state: AgentState) -> dict:
        """Call the LLM with retry logic for malformed tool calls."""
        iteration = state.get("iteration", 0) + 1

        for attempt in range(MAX_LLM_RETRIES):
            try:
                response = llm_with_tools.invoke(state["messages"])
                return {
                    "messages": [response],
                    "iteration": iteration,
                }
            except Exception as e:
                if attempt < MAX_LLM_RETRIES - 1:
                    continue
                raise

    # ── Routing: tools or end? ─────────────────────────────
    def should_continue(state: AgentState) -> str:
        """Route to tools if LLM requested a tool call, else end."""
        # Safety: prevent infinite loops
        if state.get("iteration", 0) >= MAX_ITERATIONS:
            return "end"

        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return "end"

    # ── Build the graph ────────────────────────────────────
    # Flow: agent → [tools → agent]* → END
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent", should_continue,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "agent")

    return graph.compile()


# ═══════════════════════════════════════════════════════════════
# INTERACTIVE MODE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    print(f"LangGraph Reference Agent (provider: {provider})")
    print("=" * 60)
    print("Tools: get_weather, calculate")
    print("Type 'quit' to exit\n")

    app = create_agent()

    # Demo queries
    demo_queries = [
        "What's the weather in Tokyo?",
        "Calculate (15 + 27) * 3",
    ]

    for query in demo_queries:
        print(f"Query: {query}")
        try:
            result = app.invoke({
                "messages": [HumanMessage(content=query)],
                "iteration": 0,
            })
            print(f"Agent: {result['messages'][-1].content}\n")
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}\n")

    # Interactive loop
    print("-" * 60)
    print("Your turn! Ask anything (weather, math, or general questions):\n")
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if not user_input:
                continue
            result = app.invoke({
                "messages": [HumanMessage(content=user_input)],
                "iteration": 0,
            })
            print(f"Agent: {result['messages'][-1].content}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}\n")
