"""
Example 4: ADK Tool Agent — Tools Without the Plumbing
=========================================================
ADK makes tool integration effortless. Unlike LangGraph where you
build the agent→tools→agent loop manually, ADK handles it automatically.

You just:
  1. Write plain Python functions (no decorators!)
  2. Pass them to LlmAgent(tools=[...])
  3. ADK reads the function name, docstring, and type hints to
     generate the tool schema for the LLM

This example uses the SAME tools as Example 02 (LangGraph) so you
can directly compare the two approaches.

Run: python week-02-framework-basics/examples/example_04_adk_tool_agent.py
"""

import asyncio
import logging
import os
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

logging.getLogger("google_genai.types").setLevel(logging.ERROR)

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


# ── Step 1: Define tools as plain functions ─────────────────
# In ADK, tools are just regular Python functions.
# ADK reads the function signature and docstring to tell the LLM
# what each tool does. No @tool decorator needed!
#
# IMPORTANT: Type hints and docstrings are REQUIRED — ADK uses them
# to generate the tool schema that the LLM sees.

def calculate(expression: str) -> str:
    """Evaluate a math expression and return the result.
    Use this for any arithmetic: addition, subtraction, multiplication,
    division, exponents (use **), etc.
    Example inputs: '15 * 7', '2 ** 10', '100 / 3'
    """
    try:
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return f"Error: Invalid characters in expression. Use only numbers and +-*/()"
        result = eval(expression)
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return f"Error: Division by zero in '{expression}'"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


def string_reverse(text: str) -> str:
    """Reverse a string. Use this when asked to reverse text or words.
    Input: the text to reverse. Output: the reversed text.
    """
    return text[::-1]


def word_count(text: str) -> str:
    """Count the number of words and characters in a text.
    Returns both the word count and character count.
    """
    words = len(text.split())
    chars = len(text)
    return f"'{text}' has {words} words and {chars} characters"


# ── Step 2: Create the agent with tools ─────────────────────
# Just pass the functions in the tools list. ADK does the rest.
# Compare this with LangGraph where you needed:
#   1. @tool decorator on each function
#   2. llm.bind_tools(tools)
#   3. ToolNode(tools)
#   4. Manual graph with conditional edges

tool_agent = LlmAgent(
    name="tool_agent",
    model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
    instruction="""You are a helpful assistant with access to tools.
    Use the calculate tool for math, string_reverse to reverse text,
    and word_count to count words/characters.
    Always show the tool results in your final answer.""",
    tools=[calculate, string_reverse, word_count],  # Just pass the functions!
)


# ── Step 3: Set up Runner and SessionService ────────────────

session_service = InMemorySessionService()
runner = Runner(
    agent=tool_agent,
    app_name="tool_demo",
    session_service=session_service,
)


# ══════════════════════════════════════════════════════════════
# PART 1: Basic Tool Calls — See the hidden loop
# ══════════════════════════════════════════════════════════════

async def demo_basic_tool_calls():
    """Run queries and show ALL events (not just the final response)."""
    print("=" * 60)
    print("PART 1: Basic Tool Calls (showing all events)")
    print("=" * 60)

    session = await session_service.create_session(
        app_name="tool_demo", user_id="user1"
    )

    query = "What is 15 * 7?"
    print(f"\nQuery: {query}\n")
    print("Events (what happens behind the scenes):")

    # Iterate over ALL events — this reveals the hidden tool-calling loop
    event_num = 0
    async for event in runner.run_async(
        user_id="user1",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=query)]),
    ):
        event_num += 1

        # Check what kind of event this is
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    # The LLM is requesting a tool call
                    fc = part.function_call
                    print(f"  Event {event_num}: TOOL CALL -> {fc.name}({fc.args})")
                elif hasattr(part, "function_response") and part.function_response:
                    # A tool has returned its result
                    fr = part.function_response
                    print(f"  Event {event_num}: TOOL RESULT <- {fr.name}: {fr.response}")
                elif hasattr(part, "text") and part.text:
                    if event.is_final_response():
                        print(f"  Event {event_num}: FINAL ANSWER -> {part.text}")
                    else:
                        print(f"  Event {event_num}: TEXT -> {part.text}")

    # KEY INSIGHT: ADK automatically ran the loop:
    # User query → LLM decides to call calculate → tool runs →
    # result sent back to LLM → LLM gives final answer
    print(f"\n  [ADK handled the entire tool loop automatically!]")
    print(f"  [In LangGraph, you build this loop manually with graph edges]")
    print()


# ══════════════════════════════════════════════════════════════
# PART 2: Multi-Tool Queries — Chaining tool calls
# ══════════════════════════════════════════════════════════════

async def demo_multi_tool():
    """Show that ADK can chain multiple tool calls for complex queries."""
    print("=" * 60)
    print("PART 2: Multi-Tool Queries")
    print("=" * 60)

    queries = [
        "Reverse the word 'framework' and tell me how many characters it has",
        "Calculate 2 to the power of 10, then reverse that number as a string",
    ]

    for query in queries:
        session = await session_service.create_session(
            app_name="tool_demo", user_id="user1"
        )
        print(f"\nQuery: {query}")

        async for event in runner.run_async(
            user_id="user1",
            session_id=session.id,
            new_message=types.Content(
                role="user", parts=[types.Part(text=query)]
            ),
        ):
            if event.is_final_response():
                print(f"Agent: {event.content.parts[0].text}")
    print()


# ══════════════════════════════════════════════════════════════
# PART 3: Session Persistence — Follow-up queries
# ══════════════════════════════════════════════════════════════

async def demo_session_persistence():
    """Show that tool results persist in session memory."""
    print("=" * 60)
    print("PART 3: Session Persistence (follow-up queries)")
    print("=" * 60)

    # Create ONE session for multiple turns
    session = await session_service.create_session(
        app_name="tool_demo", user_id="user1"
    )

    turns = [
        "Calculate 123 * 456",
        "Now reverse the result you just calculated as a string",
        "How many characters does that reversed string have?",
    ]

    for turn_num, query in enumerate(turns, 1):
        print(f"\n  Turn {turn_num}: {query}")

        async for event in runner.run_async(
            user_id="user1",
            session_id=session.id,  # Same session across all turns
            new_message=types.Content(
                role="user", parts=[types.Part(text=query)]
            ),
        ):
            if event.is_final_response():
                print(f"  Agent: {event.content.parts[0].text}")

    print(f"\n  [All 3 turns shared the same session -- agent remembered previous results!]")
    print()


# ══════════════════════════════════════════════════════════════
# Run all demos
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\nADK Tool Agent -- Tools Without the Plumbing")
    print("=" * 60)
    print(f"Model: {os.getenv('GOOGLE_MODEL', 'gemini-2.0-flash')}")
    print(f"Tools: calculate, string_reverse, word_count\n")

    async def run_all():
        await demo_basic_tool_calls()
        await demo_multi_tool()
        await demo_session_persistence()

    asyncio.run(run_all())

    print("=" * 60)
    print("KEY TAKEAWAYS:")
    print("  1. ADK tools are plain functions -- no decorators needed")
    print("  2. ADK reads type hints + docstrings to create tool schemas")
    print("  3. The tool-calling loop is automatic (no graph edges)")
    print("  4. Events reveal what happens behind the scenes")
    print("  5. Session persistence means follow-up queries just work")
    print("  6. Trade-off: less control than LangGraph, but much simpler")
    print("=" * 60)
