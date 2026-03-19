"""
Example 14: Why Learn Frameworks Before Patterns?
===================================================
This is a conceptual example that explains WHY the curriculum
teaches frameworks (LangGraph, ADK) in Week 2, before jumping
into advanced patterns (ReAct, multi-agent, etc.) in Weeks 3-4.

Understanding this helps you learn more effectively.

No API keys needed -- this is a teaching example with demonstrations.

Run: python week-02-framework-basics/examples/example_14_why_frameworks.py
"""


def main():
    print("\nWhy Learn Frameworks Before Patterns?")
    print("=" * 60)

    # ================================================================
    # PART 1: The Problem with Raw LLM Calls
    # ================================================================
    print("""
PART 1: The Problem with Raw LLM Calls
========================================

In Week 1, you made direct LLM calls:

    response = llm.invoke("What is AI?")

This works for simple questions, but real agents need:
  - Tool calling (search, calculate, APIs)
  - Loops (try a tool, check result, try again)
  - State (remember what happened 3 steps ago)
  - Error recovery (retry, fallback, graceful degradation)
  - Observability (what did the agent do and why?)

Building all of this from scratch is painful:""")

    # Show what "from scratch" looks like
    print("""
  # WITHOUT a framework (pseudocode):
  messages = []
  while True:
      response = llm.invoke(messages)
      if has_tool_calls(response):
          for tool_call in response.tool_calls:
              try:
                  result = execute_tool(tool_call)
                  messages.append(tool_result(result))
              except Exception:
                  messages.append(tool_error(...))
          messages.append(response)
          if len(messages) > 20:  # safety
              break
      else:
          print(response.content)
          break

  # That's 15+ lines for a BASIC tool loop.
  # Add state, error handling, logging, and it becomes 100+ lines
  # of boilerplate that's the SAME for every agent.
""")

    # ================================================================
    # PART 2: What Frameworks Give You
    # ================================================================
    print("""
PART 2: What Frameworks Give You
==================================

Frameworks solve the boilerplate problem:

  LangGraph gives you:
  + StateGraph -- visual, explicit control flow
  + ToolNode -- automatic tool execution
  + add_messages -- conversation history management
  + Conditional edges -- routing logic
  + Checkpointing -- save/resume agent state (Week 6)

  ADK gives you:
  + LlmAgent -- declarative agent config
  + Automatic tool loop -- no manual wiring
  + Session management -- built-in memory
  + SequentialAgent, AgentTool -- multi-agent patterns
  + Google Cloud deployment -- production-ready
""")

    # ================================================================
    # PART 3: Why Frameworks BEFORE Patterns
    # ================================================================
    print("""
PART 3: Why Frameworks BEFORE Patterns
========================================

The curriculum order is intentional:

  Week 1: Fundamentals (raw LLM calls, basic tools)
      |
      v  "Now you know the pain of doing it manually"
      |
  Week 2: Frameworks (LangGraph, ADK)         <-- YOU ARE HERE
      |
      v  "Now you have the tools to build complex agents"
      |
  Week 3: Basic Patterns (ReAct, chain-of-thought, HITL)
      |
      v  "Patterns are IMPLEMENTED using frameworks"
      |
  Week 4: Advanced Patterns (multi-agent, planning)

  The key insight:

    PATTERNS are WHAT your agent does (strategy)
    FRAMEWORKS are HOW you build it (implementation)

  You need the HOW before you can build the WHAT.

  Example: The ReAct pattern (Week 3) is:
    "Reason about what tool to call, Act by calling it, Observe the result"

  In LangGraph, ReAct is:
    agent_node -> should_continue -> tools_node -> agent_node (loop)

  In ADK, ReAct is:
    LlmAgent(tools=[...])  # ADK does it automatically!

  Without knowing the frameworks, you can't implement the patterns.
""")

    # ================================================================
    # PART 4: The Learning Path Ahead
    # ================================================================
    print("""
PART 4: What's Coming Next
============================

  Week 3 (Basic Patterns) -- now you can BUILD these:
    - ReAct pattern (reason + act loop)
    - Chain-of-thought (step-by-step reasoning)
    - Human-in-the-loop (ask user for confirmation)
    - Tool use patterns (when to call which tool)

  Week 4 (Advanced Patterns):
    - Multi-agent systems (agents talking to agents)
    - Planning and decomposition
    - Failure recovery patterns
    - Middleware for cross-cutting concerns

  Week 5 (Context & Memory):
    - RAG (retrieval-augmented generation)
    - Long-term memory
    - Context engineering

  Every pattern from Week 3+ is built ON TOP of what you
  learned in Week 2. The framework is your foundation.
""")

    print("=" * 60)
    print("KEY TAKEAWAYS:")
    print("  1. Raw LLM calls = too much boilerplate for real agents")
    print("  2. Frameworks handle the plumbing (loops, state, tools)")
    print("  3. Patterns are strategies; frameworks are implementations")
    print("  4. You need the HOW (frameworks) before the WHAT (patterns)")
    print("  5. Everything from Week 3+ builds on Week 2's foundation")
    print("=" * 60)


if __name__ == "__main__":
    main()
