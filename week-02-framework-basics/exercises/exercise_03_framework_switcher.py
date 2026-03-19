"""
Exercise 3: Framework Switcher
================================
Difficulty: Intermediate | Time: 2.5 hours

Task:
Build an agent system where the SAME tools and SAME query can run
on either LangGraph or ADK, switchable via a parameter. Then compare
the outputs from both frameworks.

This teaches you to think about tools as framework-agnostic logic.

Instructions:
1. Complete the shared tool functions (plain Python — no decorators)
2. Implement build_langgraph_agent() that wraps tools for LangGraph
3. Implement build_adk_agent() that uses tools with ADK
4. Implement run_with_framework() that dispatches to the right framework
5. Implement compare_frameworks() that runs both and prints results
6. Test with the queries below

Hints:
- Write tool logic ONCE as plain functions
- For LangGraph: use @tool decorator when wrapping
- For ADK: pass plain functions directly
- Use time.time() to measure execution time
- Look at example_05_framework_comparison.py for the pattern

Run: python week-02-framework-basics/exercises/exercise_03_framework_switcher.py
"""

import asyncio
import os
import time
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()


# ══════════════════════════════════════════════════════════════
# Step 1: Shared tool logic (framework-agnostic)
# ══════════════════════════════════════════════════════════════
# Write the core logic here. These functions will be wrapped
# differently for each framework.

def calculate_logic(expression: str) -> str:
    """Evaluate a math expression safely.

    Args:
        expression: A math expression like '15 * 7' or '2 ** 10'

    Returns:
        A string with the result, e.g., '15 * 7 = 105'
    """
    # TODO: Implement safe math evaluation
    # 1. Validate that only safe characters are used (digits, operators, spaces, dots, parens)
    # 2. Use eval() to compute
    # 3. Return formatted result string
    # 4. Handle errors gracefully (return error message)
    pass


def reverse_text_logic(text: str) -> str:
    """Reverse a string.

    Args:
        text: The text to reverse

    Returns:
        The reversed text
    """
    # TODO: Implement string reversal
    pass


# ══════════════════════════════════════════════════════════════
# Step 2: Build LangGraph agent
# ══════════════════════════════════════════════════════════════

def build_langgraph_agent():
    """Build a LangGraph agent using the shared tools.

    Returns:
        A compiled LangGraph app ready to invoke.
    """
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    from typing import TypedDict, Annotated
    from langgraph.graph import add_messages

    # TODO: Wrap shared functions with @tool decorator
    # @tool
    # def calculate(expression: str) -> str:
    #     """Evaluate a math expression. Example: '15 * 7'"""
    #     return calculate_logic(expression)
    #
    # @tool
    # def reverse_text(text: str) -> str:
    #     """Reverse a string."""
    #     return reverse_text_logic(text)

    # TODO: Set up LLM with provider flexibility (groq/openai)
    # TODO: Bind tools to LLM
    # TODO: Define AgentState (TypedDict with messages)
    # TODO: Define agent_node and should_continue functions
    # TODO: Build StateGraph with agent → tools → agent loop
    # TODO: Return compiled graph
    pass


# ══════════════════════════════════════════════════════════════
# Step 3: Build ADK agent
# ══════════════════════════════════════════════════════════════

def build_adk_agent():
    """Build an ADK agent using the shared tools.

    Returns:
        Tuple of (runner, session_service)
    """
    from google.adk.agents import LlmAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    # TODO: Create ADK-compatible tool functions
    # These can call the shared logic directly:
    # def calculate(expression: str) -> str:
    #     """Evaluate a math expression. Example: '15 * 7'"""
    #     return calculate_logic(expression)
    #
    # def reverse_text(text: str) -> str:
    #     """Reverse a string."""
    #     return reverse_text_logic(text)

    # TODO: Create LlmAgent with tools
    # TODO: Create InMemorySessionService and Runner
    # TODO: Return (runner, session_service)
    pass


# ══════════════════════════════════════════════════════════════
# Step 4: Run with either framework
# ══════════════════════════════════════════════════════════════

def run_with_framework(framework: str, query: str) -> dict:
    """Run a query using the specified framework.

    Args:
        framework: "langgraph" or "adk"
        query: The user's question

    Returns:
        Dict with keys: "result" (str), "time_seconds" (float), "framework" (str)
    """
    # TODO: Implement this dispatcher
    # 1. If framework == "langgraph":
    #    - Call build_langgraph_agent()
    #    - Invoke with the query
    #    - Time the execution
    #    - Return result dict
    #
    # 2. If framework == "adk":
    #    - Call build_adk_agent()
    #    - Create session, run query async
    #    - Time the execution
    #    - Return result dict
    #
    # 3. Handle errors — return error message in result dict
    pass


# ══════════════════════════════════════════════════════════════
# Step 5: Compare both frameworks
# ══════════════════════════════════════════════════════════════

def compare_frameworks(query: str):
    """Run the same query on both frameworks and print comparison.

    Args:
        query: The question to ask both frameworks
    """
    # TODO: Implement comparison
    # 1. Run with LangGraph: run_with_framework("langgraph", query)
    # 2. Run with ADK: run_with_framework("adk", query)
    # 3. Print both results side by side
    # 4. Print execution times
    # 5. Note any differences in the answers
    pass


# ══════════════════════════════════════════════════════════════
# Test your implementation
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Exercise 3: Framework Switcher")
    print("=" * 50)

    # Test 1: Simple calculation
    print("\nTest 1: What is 15 * 7?")
    # compare_frameworks("What is 15 * 7?")

    # Test 2: String operation
    print("\nTest 2: Reverse the word 'framework'")
    # compare_frameworks("Reverse the word 'framework'")

    # Test 3: Multi-step (requires chaining)
    print("\nTest 3: Calculate 2 ** 10 and then reverse that number")
    # compare_frameworks("Calculate 2 to the power of 10, then reverse that number as a string")

    print("\n(Uncomment the test code above after implementing!)")
