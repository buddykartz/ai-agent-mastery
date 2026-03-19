"""
Example 2: LangGraph Tool Agent — The ReAct Loop
===================================================
Now we add an LLM to the graph! This example builds a complete
tool-using agent that follows the ReAct (Reason + Act) pattern:

  1. LLM receives a question
  2. LLM decides: call a tool OR respond directly
  3. If tool called → execute tool → send result back to LLM → repeat
  4. If no tool call → we're done, return the response

This is THE core pattern for LangGraph agents. Master this and you
can build almost any agent.

New concepts beyond Week 1:
  - Tool call count tracking (safety valve against infinite loops)
  - Multiple tools with different capabilities
  - The full agent → tools → agent loop explained step by step

Run: python week-02-framework-basics/examples/example_02_langgraph_tool_agent.py
"""

import os
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
from langgraph.graph import add_messages


# ── Step 1: Define tools ────────────────────────────────────
# Tools are functions the LLM can call. In LangGraph, we use the
# @tool decorator which creates a schema from the function signature
# and docstring. The LLM reads this schema to decide when to call each tool.

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression and return the result.
    Use this for any arithmetic: addition, subtraction, multiplication,
    division, exponents (use **), etc.
    Example inputs: '15 * 7', '2 ** 10', '100 / 3'
    """
    try:
        # Only allow safe math characters (no imports, no exec)
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return f"Error: Invalid characters in expression. Use only numbers and +-*/()"
        result = eval(expression)
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return f"Error: Division by zero in '{expression}'"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@tool
def string_reverse(text: str) -> str:
    """Reverse a string. Use this when asked to reverse text or words.
    Input: the text to reverse. Output: the reversed text.
    """
    return text[::-1]


@tool
def word_count(text: str) -> str:
    """Count the number of words in a text.
    Also provides character count for additional detail.
    """
    words = len(text.split())
    chars = len(text)
    return f"'{text}' has {words} words and {chars} characters"


# ── Step 2: Set up the LLM with tools bound ────────────────
# bind_tools() tells the LLM about our tools so it can call them.
# The LLM will see each tool's name, description, and parameter schema.

tools = [calculate, string_reverse, word_count]

provider = os.getenv("LLM_PROVIDER", "groq").lower()
if provider == "groq":
    from langchain_groq import ChatGroq
    llm = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
else:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

# bind_tools creates a new LLM instance that knows about our tools
llm_with_tools = llm.bind_tools(tools)


# ── Step 3: Define the agent state ─────────────────────────
# We track messages (the conversation) and tool_call_count (safety valve).

class AgentState(TypedDict):
    # add_messages reducer: appends new messages instead of replacing
    messages: Annotated[list, add_messages]
    # Safety counter — prevents infinite tool-calling loops
    tool_call_count: int


# ── Step 4: Define the agent node ──────────────────────────
# The agent node calls the LLM and returns its response.
# The LLM's response may contain tool_calls (if it wants to use a tool)
# or just text (if it has the answer).

def agent_node(state: AgentState) -> dict:
    """Call the LLM. It will either request a tool call or give a final answer."""
    # Groq/Llama can sometimes generate malformed tool calls.
    # Retry up to 3 times to handle these transient errors.
    for attempt in range(3):
        try:
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}
        except Exception as e:
            if attempt < 2 and ("tool_use_failed" in str(e) or "malformed" in str(e).lower()):
                print(f"  [Retry {attempt + 1}/3] Malformed tool call, retrying...")
                continue
            raise


# ── Step 5: Define the routing function ─────────────────────
# After the LLM responds, we check: did it request a tool call?
# If yes → route to tools. If no → we're done (route to END).
# Also: if we've called tools too many times, force-stop to prevent loops.

MAX_TOOL_CALLS = 5  # Safety limit

def should_continue(state: AgentState) -> str:
    """Decide whether to call tools, or end the conversation.

    Returns:
        'tools' — the LLM wants to call a tool
        'end'   — the LLM gave a final answer (or we hit the safety limit)
    """
    last_message = state["messages"][-1]

    # Safety valve: if we've called tools too many times, force stop
    if state.get("tool_call_count", 0) >= MAX_TOOL_CALLS:
        print(f"  [Safety] Hit max tool calls ({MAX_TOOL_CALLS}). Forcing end.")
        return "end"

    # Check if the LLM's response contains tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # No tool calls → LLM gave a final text answer
    return "end"


# ── Step 6: Tool call counter node ─────────────────────────
# We wrap the ToolNode to also increment our safety counter.

tool_executor = ToolNode(tools)

def tools_node(state: AgentState) -> dict:
    """Execute tool calls and increment the counter."""
    # Run the actual tools via LangGraph's built-in ToolNode
    result = tool_executor.invoke(state)
    # Count how many tool calls we've made
    new_count = state.get("tool_call_count", 0) + len(state["messages"][-1].tool_calls)
    return {
        "messages": result["messages"],
        "tool_call_count": new_count,
    }


# ── Step 7: Build the complete graph ───────────────────────
# The graph structure:
#
#   ┌──────────┐    has tool calls    ┌───────────┐
#   │  agent   │ ──────────────────→  │   tools   │
#   │  (LLM)   │ ←────────────────── │ (execute)  │
#   └──────────┘    tool results      └───────────┘
#        │
#        │ no tool calls (final answer)
#        ▼
#      [END]

graph = StateGraph(AgentState)

# Add nodes
graph.add_node("agent", agent_node)
graph.add_node("tools", tools_node)

# Set entry point — agent always goes first
graph.set_entry_point("agent")

# Add conditional edges from agent
graph.add_conditional_edges(
    "agent",          # After the agent node...
    should_continue,  # ...run this function to decide where to go
    {
        "tools": "tools",  # If "tools" → go to tools node
        "end": END,         # If "end" → stop the graph
    },
)

# After tools run, always go back to agent (so it can process results)
graph.add_edge("tools", "agent")

# Compile into a runnable app
app = graph.compile()


# ── Step 8: Run it! ────────────────────────────────────────

if __name__ == "__main__":
    print(f"LangGraph Tool Agent (provider: {provider})")
    print("=" * 60)
    print("Tools available: calculate, string_reverse, word_count\n")

    # Test queries of increasing complexity
    queries = [
        # Simple: single tool call
        "What is 15 * 7?",
        # Medium: requires two tool calls
        "Reverse the word 'framework' and tell me how many characters it has",
        # Complex: chain of tool calls
        "Calculate 2 to the power of 10, then reverse that number as a string",
    ]

    for i, query in enumerate(queries, 1):
        print(f"{'-' * 60}")
        print(f"Query {i}: {query}")
        print(f"{'-' * 60}")
        try:
            result = app.invoke({
                "messages": [HumanMessage(content=query)],
                "tool_call_count": 0,
            })

            # The last message is the agent's final answer
            final_answer = result["messages"][-1].content
            tool_calls_made = result.get("tool_call_count", 0)

            print(f"Agent: {final_answer}")
            print(f"  [Tool calls made: {tool_calls_made}]")
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")
        print()

    print("=" * 60)
    print("KEY TAKEAWAYS:")
    print("  1. The agent -> tools -> agent loop is the core ReAct pattern")
    print("  2. The LLM decides WHICH tool to call based on the question")
    print("  3. should_continue() is the routing function -- tools or end")
    print("  4. Always add a safety valve (max tool calls) to prevent loops")
    print("  5. bind_tools() tells the LLM about available tools")
    print("=" * 60)
