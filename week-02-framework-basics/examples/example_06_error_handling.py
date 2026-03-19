"""
Example 6: Error Handling — Making Agents Resilient
=====================================================
Real-world agents face many error scenarios:
  - API timeouts and rate limits
  - Malformed tool calls (common with Groq/Llama)
  - Tool execution failures
  - Unexpected LLM responses

This example teaches systematic error handling patterns for both
LangGraph and ADK, so your agents degrade gracefully instead of crashing.

Run: python week-02-framework-basics/examples/example_06_error_handling.py
"""

import logging
import os
import time
import random
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

logging.getLogger("google_genai.types").setLevel(logging.ERROR)


# ══════════════════════════════════════════════════════════════
# PATTERN 1: Retry with Exponential Backoff
# ══════════════════════════════════════════════════════════════
# When an API call fails, wait progressively longer between retries.
# This avoids hammering the API and respects rate limits.

def retry_with_backoff(func, max_retries=3, base_delay=1.0):
    """Retry a function with exponential backoff.

    Args:
        func: A callable to retry
        max_retries: Maximum number of attempts
        base_delay: Initial delay in seconds (doubles each retry)

    Returns:
        The function's return value on success

    Raises:
        The last exception if all retries fail
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                # Last attempt — re-raise the error
                raise
            # Calculate delay: 1s, 2s, 4s, 8s, ...
            delay = base_delay * (2 ** attempt)
            print(f"  [Retry {attempt + 1}/{max_retries}] {type(e).__name__}: {e}")
            print(f"  [Waiting {delay:.1f}s before retry...]")
            time.sleep(delay)


def demo_retry_pattern():
    """Show the retry pattern with a simulated flaky function."""
    print("=" * 60)
    print("PATTERN 1: Retry with Exponential Backoff")
    print("=" * 60)

    # Simulate a function that fails 2 times then succeeds
    call_count = 0

    def flaky_api_call():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise ConnectionError(f"API timeout (attempt {call_count})")
        return "Success! Got the response."

    print("\nSimulating a flaky API call (fails twice, then succeeds):\n")
    try:
        result = retry_with_backoff(flaky_api_call, max_retries=3, base_delay=0.1)
        print(f"\n  Result: {result}")
        print(f"  Total attempts: {call_count}")
    except Exception as e:
        print(f"\n  Failed after all retries: {e}")
    print()


# ══════════════════════════════════════════════════════════════
# PATTERN 2: Tool Execution Error Handling
# ══════════════════════════════════════════════════════════════
# Tools can fail — network errors, invalid input, etc.
# The key: return an error MESSAGE instead of raising an exception.
# This lets the LLM read the error and try a different approach.

def demo_tool_error_handling():
    """Show how to handle tool failures gracefully."""
    print("=" * 60)
    print("PATTERN 2: Tool Error Handling (return errors, don't crash)")
    print("=" * 60)

    # BAD: This tool crashes on bad input
    def bad_divide(a: float, b: float) -> float:
        return a / b  # Crashes on b=0!

    # GOOD: This tool returns an error message
    def safe_divide(a: float, b: float) -> str:
        """Divide two numbers safely."""
        try:
            if b == 0:
                return f"Error: Cannot divide {a} by zero. Please provide a non-zero divisor."
            result = a / b
            return f"{a} / {b} = {result:.4f}"
        except Exception as e:
            return f"Error: Could not compute {a}/{b}: {e}"

    print("\nBAD tool (crashes on division by zero):")
    try:
        result = bad_divide(10, 0)
    except ZeroDivisionError as e:
        print(f"  CRASH! {type(e).__name__}: {e}")
        print(f"  -> The agent would crash entirely!")

    print("\nGOOD tool (returns error message):")
    result = safe_divide(10, 0)
    print(f"  Result: {result}")
    print(f"  -> The LLM receives this message and can tell the user what happened")

    print(f"\nGOOD tool (normal operation):")
    result = safe_divide(10, 3)
    print(f"  Result: {result}")
    print()


# ══════════════════════════════════════════════════════════════
# PATTERN 3: LangGraph Agent with Error Recovery
# ══════════════════════════════════════════════════════════════
# Build a LangGraph agent that handles errors at every level:
#   1. LLM call errors (retry)
#   2. Tool execution errors (safe return)
#   3. Overall timeout (max iterations)
#   4. Fallback response on complete failure

def demo_langgraph_error_recovery():
    """Full LangGraph agent with comprehensive error handling."""
    print("=" * 60)
    print("PATTERN 3: LangGraph Agent with Error Recovery")
    print("=" * 60)

    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage, AIMessage
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    from typing import TypedDict, Annotated
    from langgraph.graph import add_messages

    # ── Flaky tool — fails randomly to simulate real-world issues ──
    @tool
    def unreliable_lookup(query: str) -> str:
        """Look up information about a topic. May occasionally fail due to network issues."""
        # Simulate 40% failure rate
        if random.random() < 0.4:
            return f"Error: Service temporarily unavailable. Try rephrasing your query."
        return f"Found info about '{query}': This is a reliable result about {query}."

    @tool
    def always_works(query: str) -> str:
        """A reliable fallback tool for basic information."""
        return f"Basic info about '{query}': {query} is an interesting topic with many aspects."

    tools = [unreliable_lookup, always_works]

    # ── LLM setup ──────────────────────────────────────────
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    else:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    llm_with_tools = llm.bind_tools(tools)

    # ── State with error tracking ──────────────────────────
    class ErrorAwareState(TypedDict):
        messages: Annotated[list, add_messages]
        error_count: int       # How many errors we've encountered
        iteration_count: int   # Total loop iterations

    MAX_ITERATIONS = 6   # Prevent infinite loops
    MAX_ERRORS = 3        # Give up after too many errors

    # ── Agent node with retry logic ────────────────────────
    def agent_node(state: ErrorAwareState) -> dict:
        """Call the LLM with retry logic for malformed tool calls."""
        iteration = state.get("iteration_count", 0) + 1

        for attempt in range(3):
            try:
                response = llm_with_tools.invoke(state["messages"])
                return {
                    "messages": [response],
                    "iteration_count": iteration,
                }
            except Exception as e:
                if attempt < 2:
                    print(f"  [agent retry {attempt + 1}] {type(e).__name__}")
                    continue
                # After 3 attempts, return a graceful fallback
                print(f"  [agent] LLM call failed after 3 retries: {e}")
                fallback = AIMessage(content=f"I encountered an error processing your request. "
                                             f"Please try rephrasing your question.")
                return {
                    "messages": [fallback],
                    "error_count": state.get("error_count", 0) + 1,
                    "iteration_count": iteration,
                }

    # ── Routing with safety checks ─────────────────────────
    def should_continue(state: ErrorAwareState) -> str:
        """Route with multiple safety checks."""
        # Check 1: Max iterations (prevent infinite loops)
        if state.get("iteration_count", 0) >= MAX_ITERATIONS:
            print(f"  [safety] Max iterations ({MAX_ITERATIONS}) reached")
            return "end"

        # Check 2: Too many errors (give up gracefully)
        if state.get("error_count", 0) >= MAX_ERRORS:
            print(f"  [safety] Too many errors ({MAX_ERRORS})")
            return "end"

        # Check 3: Normal routing — does the LLM want a tool?
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return "end"

    # ── Build the error-resilient graph ────────────────────
    graph = StateGraph(ErrorAwareState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent", should_continue,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "agent")
    app = graph.compile()

    # ── Run it ─────────────────────────────────────────────
    print(f"\nProvider: {provider}")
    print(f"Safety limits: max_iterations={MAX_ITERATIONS}, max_errors={MAX_ERRORS}\n")

    query = "Look up information about machine learning"
    print(f"Query: {query}\n")

    # Set seed for reproducible demo
    random.seed(42)

    try:
        result = app.invoke({
            "messages": [HumanMessage(content=query)],
            "error_count": 0,
            "iteration_count": 0,
        })
        print(f"\nAgent: {result['messages'][-1].content}")
        print(f"  [Iterations: {result['iteration_count']}, Errors: {result['error_count']}]")
    except Exception as e:
        print(f"\nFatal error: {type(e).__name__}: {e}")
        print(f"  -> In production, return a user-friendly error message here")
    print()


# ══════════════════════════════════════════════════════════════
# PATTERN 4: ADK Error Handling
# ══════════════════════════════════════════════════════════════

def demo_adk_error_handling():
    """Show error handling patterns for ADK agents."""
    print("=" * 60)
    print("PATTERN 4: ADK Error Handling")
    print("=" * 60)

    import asyncio
    from google.adk.agents import LlmAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    # ── Safe ADK tool ──────────────────────────────────────
    def safe_lookup(query: str) -> str:
        """Look up information. Returns error messages instead of crashing."""
        try:
            if not query or len(query.strip()) == 0:
                return "Error: Please provide a non-empty search query."
            if len(query) > 500:
                return "Error: Query too long. Please use fewer than 500 characters."
            # Simulate occasional failure
            if random.random() < 0.3:
                return f"Error: Service temporarily unavailable for '{query}'. Please try again."
            return f"Found: '{query}' is an important topic in technology."
        except Exception as e:
            # Catch-all: NEVER let a tool crash
            return f"Error: Unexpected issue processing '{query}': {e}"

    # ── Agent with error-aware instruction ─────────────────
    agent = LlmAgent(
        name="resilient_agent",
        model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
        instruction="""You are a helpful research assistant.
        Use the safe_lookup tool to find information.

        IMPORTANT error handling rules:
        - If a tool returns an error message, acknowledge it and try a different approach
        - If a tool fails repeatedly (3+ times), provide a helpful response based on your knowledge
        - Never claim you found specific data if the tool returned an error
        - Be transparent about limitations""",
        tools=[safe_lookup],
    )

    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="resilient_app", session_service=session_service)

    async def run_with_error_handling():
        session = await session_service.create_session(
            app_name="resilient_app", user_id="user1"
        )

        query = "Tell me about quantum computing"
        print(f"\nQuery: {query}")

        random.seed(99)

        try:
            async for event in runner.run_async(
                user_id="user1",
                session_id=session.id,
                new_message=types.Content(
                    role="user", parts=[types.Part(text=query)]
                ),
            ):
                if event.is_final_response():
                    print(f"Agent: {event.content.parts[0].text}")
        except Exception as e:
            # Catch any ADK-level errors
            print(f"  ADK Error: {type(e).__name__}: {e}")
            print(f"  -> In production, return a fallback response to the user")

    print(f"\nModel: {os.getenv('GOOGLE_MODEL', 'gemini-2.0-flash')}")
    asyncio.run(run_with_error_handling())
    print()


# ══════════════════════════════════════════════════════════════
# PATTERN 5: Error Handling Best Practices Summary
# ══════════════════════════════════════════════════════════════

def print_best_practices():
    """Print a summary of error handling best practices."""
    print("=" * 60)
    print("ERROR HANDLING BEST PRACTICES")
    print("=" * 60)
    print("""
  1. TOOLS: Return error messages, never raise exceptions
     Bad:  raise ValueError("Division by zero")
     Good: return "Error: Cannot divide by zero"
     Why:  The LLM can read the error and try a different approach

  2. LLM CALLS: Retry with exponential backoff
     - Start with 1s delay, double each time (1s, 2s, 4s)
     - Max 3 retries for transient errors
     - Groq/Llama: also retry on malformed tool calls

  3. LOOPS: Always set a maximum iteration count
     - Prevent infinite tool-calling loops
     - 5-10 iterations is usually enough
     - Force-end with a helpful message when limit is reached

  4. GRACEFUL DEGRADATION: Always have a fallback
     - If tool fails → tell user what happened
     - If LLM fails → return a pre-written fallback response
     - If everything fails → never show raw stack traces to users

  5. ADK-SPECIFIC: Error-aware instructions
     - Tell the agent HOW to handle tool errors in the instruction
     - ADK doesn't have custom routing — the instruction IS your control

  6. LANGGRAPH-SPECIFIC: Error tracking in state
     - Add error_count to your state TypedDict
     - Use conditional edges to route to fallback nodes on error
     - You have full control over the error recovery flow
""")


# ══════════════════════════════════════════════════════════════
# Run all demos
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\nError Handling Patterns for AI Agents")
    print("=" * 60)

    demo_retry_pattern()
    demo_tool_error_handling()
    demo_langgraph_error_recovery()
    demo_adk_error_handling()
    print_best_practices()
