"""
Example 12: Basic Middleware -- Logging Agent Execution
=========================================================
Middleware sits between your code and the LLM, intercepting calls
to add cross-cutting concerns like logging, timing, and monitoring.

This example shows:
  1. Logging decorator for LangGraph nodes
  2. Timing every LLM call and tool execution
  3. Structured logging format for production debugging
  4. How to add logging to ADK agents

Middleware becomes critical in production -- you need to know what
your agent did, how long it took, and what went wrong.

Run: python week-02-framework-basics/examples/example_12_middleware_logging.py
"""

import os
import time
import logging
from functools import wraps
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

logging.getLogger("google_genai.types").setLevel(logging.ERROR)


# ================================================================
# PART 1: Setting Up Structured Logging
# ================================================================

def setup_logger(name: str = "agent") -> logging.Logger:
    """Create a structured logger for agent execution.

    Structured logs make it easy to search and filter in production.
    Format: TIMESTAMP | LEVEL | COMPONENT | MESSAGE
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)

    return logger


def demo_logging_setup():
    """Show basic structured logging."""
    print("=" * 60)
    print("PART 1: Structured Logging Setup")
    print("=" * 60)

    logger = setup_logger("demo")

    # Different log levels for different situations
    logger.info("Agent started processing query")
    logger.info("Tool call: get_weather('London')")
    logger.warning("Tool returned error, retrying...")
    logger.info("Retry successful")
    logger.info("Agent completed in 2.3s")

    print("\nLog levels explained:")
    print("  DEBUG   - Detailed internal state (disabled in production)")
    print("  INFO    - Normal operations (tool calls, completions)")
    print("  WARNING - Recoverable issues (retries, fallbacks)")
    print("  ERROR   - Failures that affect the response")
    print()


# ================================================================
# PART 2: Node Execution Logging Decorator
# ================================================================

def log_node(logger: logging.Logger):
    """Decorator that logs LangGraph node execution.

    Wraps a node function to automatically log:
    - When the node starts (with state info)
    - When it completes (with timing)
    - If it fails (with error details)

    Usage:
        @log_node(logger)
        def my_node(state):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(state, *args, **kwargs):
            node_name = func.__name__
            iteration = state.get("iteration", "?")
            logger.info(f"[{node_name}] START | iteration={iteration}")
            start = time.time()

            try:
                result = func(state, *args, **kwargs)
                elapsed = time.time() - start
                logger.info(f"[{node_name}] DONE | {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start
                logger.error(f"[{node_name}] FAIL | {elapsed:.3f}s | "
                             f"{type(e).__name__}: {e}")
                raise

        return wrapper
    return decorator


def demo_logged_langgraph():
    """LangGraph agent with logging on every node."""
    print("=" * 60)
    print("PART 2: LangGraph Agent with Logging Middleware")
    print("=" * 60)

    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    from typing import TypedDict, Annotated
    from langgraph.graph import add_messages

    logger = setup_logger("langgraph_agent")

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

    tools = [calculate]

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
        iteration: int

    # The decorator adds logging to every call of this node
    @log_node(logger)
    def agent_node(state: AgentState) -> dict:
        """Call the LLM -- logging is handled by the decorator."""
        for attempt in range(3):
            try:
                response = llm_with_tools.invoke(state["messages"])
                return {
                    "messages": [response],
                    "iteration": state.get("iteration", 0) + 1,
                }
            except Exception as e:
                if attempt < 2:
                    logger.warning(f"[agent_node] Retry {attempt+1}: {e}")
                    continue
                raise

    @log_node(logger)
    def tools_node(state: AgentState) -> dict:
        """Execute tools -- logging is handled by the decorator."""
        tool_executor = ToolNode(tools)
        result = tool_executor.invoke(state)
        return {"messages": result["messages"]}

    def should_continue(state):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    app = graph.compile()

    print(f"\nRunning logged agent (provider: {provider}):\n")
    query = "What is 42 * 17?"
    logger.info(f"Query: {query}")

    try:
        result = app.invoke({
            "messages": [HumanMessage(content=query)],
            "iteration": 0,
        })
        logger.info(f"Final answer: {result['messages'][-1].content}")
    except Exception as e:
        logger.error(f"Agent failed: {e}")
    print()


# ================================================================
# PART 3: ADK Logging with Event Inspection
# ================================================================

def demo_adk_logging():
    """Show how to add logging to ADK agents via events."""
    print("=" * 60)
    print("PART 3: ADK Agent Logging via Events")
    print("=" * 60)

    import asyncio
    from google.adk.agents import LlmAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    logger = setup_logger("adk_agent")

    def calculate(expression: str) -> str:
        """Evaluate a math expression."""
        # Log tool execution
        logger.info(f"[tool:calculate] input='{expression}'")
        start = time.time()
        try:
            allowed = set("0123456789+-*/.() ")
            if not all(c in allowed for c in expression):
                result = "Error: Invalid characters"
            else:
                result = f"{expression} = {eval(expression)}"
            logger.info(f"[tool:calculate] output='{result}' ({time.time()-start:.3f}s)")
            return result
        except Exception as e:
            logger.error(f"[tool:calculate] error={e}")
            return f"Error: {e}"

    agent = LlmAgent(
        name="logged_agent",
        model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
        instruction="Use the calculate tool to solve math. Show results.",
        tools=[calculate],
    )

    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="logged_app", session_service=session_service)

    async def run_logged():
        session = await session_service.create_session(
            app_name="logged_app", user_id="user1"
        )

        query = "What is 99 * 77?"
        logger.info(f"Query: {query}")
        start = time.time()

        # In ADK, we log by inspecting ALL events (not just final)
        async for event in runner.run_async(
            user_id="user1",
            session_id=session.id,
            new_message=types.Content(
                role="user", parts=[types.Part(text=query)]
            ),
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        logger.info(f"[event] tool_call: {part.function_call.name}")
                    elif hasattr(part, "text") and part.text and event.is_final_response():
                        elapsed = time.time() - start
                        logger.info(f"[event] final_response ({elapsed:.2f}s): "
                                    f"{part.text[:80]}")

    asyncio.run(run_logged())
    print()


# ================================================================
# PART 4: Best Practices
# ================================================================

def demo_best_practices():
    """Summarize logging best practices."""
    print("=" * 60)
    print("PART 4: Logging Best Practices for Agents")
    print("=" * 60)
    print("""
  WHAT TO LOG:
    + Every LLM call (start, end, duration, token count)
    + Every tool call (name, input args, output, duration)
    + Routing decisions (which path was taken and why)
    + Errors and retries (what failed, what was the fallback)
    + Total query duration (start to final response)

  WHAT NOT TO LOG:
    - Full prompt text in production (security/privacy risk)
    - Full LLM responses (too verbose, use trace IDs instead)
    - Sensitive user data (PII, credentials)

  LOG LEVELS IN PRACTICE:
    DEBUG  -> Full prompt/response text (dev only)
    INFO   -> Tool calls, completions, timings
    WARNING -> Retries, fallbacks, slow responses
    ERROR  -> Failed calls, crashed tools, timeouts

  PRODUCTION PATTERN:
    1. Use structured logging (key=value format)
    2. Add trace IDs to correlate logs across a request
    3. Send logs to a central system (ELK, CloudWatch, etc.)
    4. Use Phoenix for LLM-specific tracing (next example!)
""")


# ================================================================
# Run all demos
# ================================================================

if __name__ == "__main__":
    print("\nBasic Middleware: Logging Agent Execution")
    print("=" * 60)

    demo_logging_setup()
    demo_logged_langgraph()
    demo_adk_logging()
    demo_best_practices()

    print("=" * 60)
    print("KEY TAKEAWAYS:")
    print("  1. Logging decorators add observability without changing logic")
    print("  2. Log every LLM call, tool call, and routing decision")
    print("  3. Structured format (timestamp|level|component|message)")
    print("  4. LangGraph: wrap nodes with decorators")
    print("  5. ADK: inspect events in the async loop")
    print("  6. Use Phoenix for deeper LLM tracing (next example)")
    print("=" * 60)
