"""
Exercise 1: Calculator Agent (LangGraph)
==========================================
Difficulty: Beginner | Time: 1.5 hours

Task:
Build a LangGraph agent with calculator tools that can solve
multi-step math problems. The agent should be able to chain
tool calls (e.g., "add 15 and 27, then multiply by 3").

Instructions:
1. Complete the 3 tool functions: add, multiply, divide
2. Handle division by zero in the divide tool (return error message!)
3. Set up the LLM with provider flexibility (groq/openai)
4. Build the StateGraph with agent_node and ToolNode
5. Add conditional edges for the agent → tools → agent loop
6. Test with all 3 queries below

Hints:
- Look at example_02_langgraph_tool_agent.py for the full pattern
- Tools should return STRINGS (not numbers) — the LLM reads them
- Division by zero should return an error message, NOT crash
- Use bind_tools() to tell the LLM about your tools

Run: python week-02-framework-basics/exercises/exercise_01_calculator_agent.py
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


# ── Step 1: Define calculator tools ─────────────────────────
# Each tool needs the @tool decorator, type hints, and a docstring.
# Return strings so the LLM can read the results.

@tool
def add(a: float, b: float) -> str:
    """Add two numbers together. Example: add(15, 27) returns '15 + 27 = 42'"""
    # TODO: Implement addition
    # Return a string like "15.0 + 27.0 = 42.0"
    pass


@tool
def multiply(a: float, b: float) -> str:
    """Multiply two numbers. Example: multiply(8, 12) returns '8 * 12 = 96'"""
    # TODO: Implement multiplication
    # Return a string like "8.0 * 12.0 = 96.0"
    pass


@tool
def divide(a: float, b: float) -> str:
    """Divide a by b. Example: divide(100, 4) returns '100 / 4 = 25.0'
    Handles division by zero gracefully.
    """
    # TODO: Implement division
    # 1. Check if b is zero — if so, return an error message string
    #    (DON'T raise an exception — return a friendly error message)
    # 2. Otherwise, return the result as a string
    pass


# ── Step 2: Set up the LLM ─────────────────────────────────
# TODO: Initialize the LLM based on the LLM_PROVIDER env variable
# Default to groq, fallback to openai
# Then bind the tools to the LLM using bind_tools()

tools = [add, multiply, divide]

# TODO: Create llm (ChatGroq or ChatOpenAI based on provider)
# llm = ...

# TODO: Bind tools to the LLM
# llm_with_tools = llm.bind_tools(tools)


# ── Step 3: Define the state ────────────────────────────────
# TODO: Create AgentState TypedDict with:
#   - messages: Annotated[list, add_messages]

# class AgentState(TypedDict):
#     ...


# ── Step 4: Define the agent node ──────────────────────────
# TODO: Create agent_node function that:
#   1. Calls llm_with_tools.invoke(state["messages"])
#   2. Returns {"messages": [response]}
#   3. (Bonus) Add retry logic for malformed tool calls

# def agent_node(state):
#     ...


# ── Step 5: Define the routing function ─────────────────────
# TODO: Create should_continue function that:
#   1. Checks if the last message has tool_calls
#   2. Returns "tools" if yes, "end" if no

# def should_continue(state):
#     ...


# ── Step 6: Build the graph ─────────────────────────────────
# TODO: Create the StateGraph:
#   1. Add "agent" node (agent_node function)
#   2. Add "tools" node (ToolNode(tools))
#   3. Set entry point to "agent"
#   4. Add conditional edges from "agent" → "tools" or END
#   5. Add edge from "tools" → "agent"
#   6. Compile the graph

# graph = StateGraph(AgentState)
# ...
# app = graph.compile()


# ── Test your implementation ────────────────────────────────

if __name__ == "__main__":
    print("Exercise 1: Calculator Agent")
    print("=" * 50)

    # Test 1: Simple addition
    print("\nTest 1: What is 15 + 27?")
    # result = app.invoke({
    #     "messages": [HumanMessage(content="What is 15 + 27?")],
    # })
    # print(f"Agent: {result['messages'][-1].content}")

    # Test 2: Multi-step calculation (requires chaining)
    print("\nTest 2: Multiply 8 by 12, then add 5 to the result")
    # result = app.invoke({
    #     "messages": [HumanMessage(content="Multiply 8 by 12, then add 5 to the result")],
    # })
    # print(f"Agent: {result['messages'][-1].content}")

    # Test 3: Division by zero (should handle gracefully!)
    print("\nTest 3: Divide 100 by 0")
    # result = app.invoke({
    #     "messages": [HumanMessage(content="Divide 100 by 0")],
    # })
    # print(f"Agent: {result['messages'][-1].content}")

    print("\n(Uncomment the test code above after implementing!)")
