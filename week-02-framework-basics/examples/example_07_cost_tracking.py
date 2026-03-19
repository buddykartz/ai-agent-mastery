"""
Example 7: Cost Tracking — Know What Your Agents Cost
=======================================================
LLM API calls cost money. Multi-tool agents make MULTIPLE LLM calls
per query (one per loop iteration), so costs add up fast.

This example shows how to:
  1. Track token usage per LLM call
  2. Use the shared CostTracker utility
  3. Set budget alerts to avoid surprises
  4. Compare costs between providers and query complexity

Uses: shared/utils/cost_tracker.py (the project's cost tracking utility)

Run: python week-02-framework-basics/examples/example_07_cost_tracking.py
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

# Add project root to path so we can import shared utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from shared.utils.cost_tracker import CostTracker


# ══════════════════════════════════════════════════════════════
# PART 1: Understanding Token Usage
# ══════════════════════════════════════════════════════════════

def demo_token_tracking():
    """Show how to extract token usage from LLM responses."""
    print("=" * 60)
    print("PART 1: Token Usage Tracking")
    print("=" * 60)

    from langchain_core.messages import HumanMessage, SystemMessage

    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
        model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    else:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    print(f"\nProvider: {provider}, Model: {model_name}\n")

    # Make a simple call and extract token usage
    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is an AI agent? Answer in one sentence."),
    ])

    print(f"Response: {response.content}\n")

    # Token usage is in response_metadata (format varies by provider)
    metadata = response.response_metadata
    usage = metadata.get("token_usage", metadata.get("usage", {}))

    # Different providers use different key names
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
    total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

    print(f"Token Usage:")
    print(f"  Input tokens:  {input_tokens:,} (your prompt)")
    print(f"  Output tokens: {output_tokens:,} (LLM's response)")
    print(f"  Total tokens:  {total_tokens:,}")
    print(f"\n  Raw metadata keys: {list(usage.keys()) if usage else 'N/A'}")
    print()

    return model_name, input_tokens, output_tokens


# ══════════════════════════════════════════════════════════════
# PART 2: Using the CostTracker
# ══════════════════════════════════════════════════════════════

def demo_cost_tracker():
    """Show how to use the shared CostTracker utility."""
    print("=" * 60)
    print("PART 2: CostTracker Utility")
    print("=" * 60)

    # Create a tracker with a $1.00 weekly budget
    tracker = CostTracker(weekly_budget=1.00)

    print(f"\nWeekly budget: ${tracker.weekly_budget:.2f}")
    print(f"Alert threshold: {tracker.alert_threshold * 100:.0f}%\n")

    # Simulate several API calls with different models
    # (In real code, you'd log actual token counts from response metadata)
    simulated_calls = [
        ("llama-3.3-70b-versatile", 150, 300),   # Groq call
        ("llama-3.3-70b-versatile", 200, 500),   # Groq call (bigger)
        ("gpt-4o-mini", 150, 300),                # OpenAI call
        ("gpt-4o-mini", 800, 1200),               # OpenAI call (bigger)
        ("gemini-3-flash-preview", 500, 1000),    # Gemini (free tier!)
    ]

    print("Logging simulated API calls:\n")
    for model, input_tok, output_tok in simulated_calls:
        cost = tracker.log_call(model, input_tok, output_tok)
        print(f"  {model:<30} in={input_tok:>5} out={output_tok:>5} cost=${cost:.6f}")

    # Print the full report
    tracker.report()
    print()


# ══════════════════════════════════════════════════════════════
# PART 3: Tracked LangGraph Agent
# ══════════════════════════════════════════════════════════════

def demo_tracked_langgraph_agent():
    """LangGraph agent with cost tracking on every LLM call."""
    print("=" * 60)
    print("PART 3: LangGraph Agent with Cost Tracking")
    print("=" * 60)

    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    from typing import TypedDict, Annotated
    from langgraph.graph import add_messages

    # Create a tracker for this demo
    tracker = CostTracker(weekly_budget=0.50)

    @tool
    def calculate(expression: str) -> str:
        """Evaluate a math expression. Example: '15 * 7'"""
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
        model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    else:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    llm_with_tools = llm.bind_tools(tools)

    class TrackedState(TypedDict):
        messages: Annotated[list, add_messages]

    # ── The key: track costs inside the agent node ─────────
    def tracked_agent_node(state: TrackedState) -> dict:
        """Agent node that logs token usage after every LLM call."""
        for attempt in range(3):
            try:
                response = llm_with_tools.invoke(state["messages"])

                # Extract token usage from response metadata
                usage = response.response_metadata.get(
                    "token_usage",
                    response.response_metadata.get("usage", {}),
                )
                input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
                output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))

                # Log the cost
                cost = tracker.log_call(model_name, input_tokens, output_tokens)
                print(f"  [cost] LLM call: {input_tokens} in + {output_tokens} out = ${cost:.6f}")

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

    graph = StateGraph(TrackedState)
    graph.add_node("agent", tracked_agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    app = graph.compile()

    # ── Run queries and see costs accumulate ───────────────
    print(f"\nProvider: {provider}, Model: {model_name}")
    print(f"Budget: ${tracker.weekly_budget:.2f}\n")

    queries = [
        "What is 15 * 7?",                    # Simple: 1-2 LLM calls
        "Calculate 100 / 4 and then 25 * 8",  # Complex: 2-3 LLM calls
    ]

    for query in queries:
        print(f"{'-' * 50}")
        print(f"Query: {query}")
        try:
            result = app.invoke({"messages": [HumanMessage(content=query)]})
            print(f"Answer: {result['messages'][-1].content}")
        except Exception as e:
            print(f"Error: {e}")
        print()

    # Show cumulative costs
    tracker.report()
    print()


# ══════════════════════════════════════════════════════════════
# PART 4: Cost Comparison Table
# ══════════════════════════════════════════════════════════════

def demo_cost_comparison():
    """Show cost differences between models for the same workload."""
    print("=" * 60)
    print("PART 4: Model Cost Comparison")
    print("=" * 60)

    from shared.utils.cost_tracker import MODEL_PRICING

    # Simulate the same workload across different models
    # Typical tool-agent query: ~200 input + ~300 output per call, 3 calls per query
    calls_per_query = 3
    input_per_call = 200
    output_per_call = 300

    print(f"\nSimulated workload per query:")
    print(f"  LLM calls:       {calls_per_query}")
    print(f"  Input tokens:    {input_per_call} per call")
    print(f"  Output tokens:   {output_per_call} per call")
    print(f"  Total tokens:    {(input_per_call + output_per_call) * calls_per_query:,}")

    print(f"\n{'Model':<35} {'Cost/Query':>12} {'Cost/100 Queries':>18} {'Free Tier':>10}")
    print(f"{'-' * 35} {'-' * 12} {'-' * 18} {'-' * 10}")

    for model, prices in sorted(MODEL_PRICING.items()):
        cost_per_query = (
            (input_per_call * prices["input"] + output_per_call * prices["output"])
            * calls_per_query / 1_000_000
        )
        cost_100 = cost_per_query * 100
        free = "Yes" if prices["input"] == 0 and prices["output"] == 0 else "No"

        print(f"  {model:<33} ${cost_per_query:.6f}   ${cost_100:.4f}           {free}")

    print(f"""
  KEY INSIGHTS:
  • Gemini Flash is free — great for learning and prototyping
  • Groq (Llama) is very affordable at ~$0.001 per query
  • GPT-4o is 10-50x more expensive than Groq
  • Tool agents multiply costs: each loop iteration = another LLM call
  • A 3-tool query costs 3x a simple question!
""")


# ══════════════════════════════════════════════════════════════
# Run all demos
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\nCost Tracking for AI Agents")
    print("=" * 60)

    demo_token_tracking()
    demo_cost_tracker()
    demo_tracked_langgraph_agent()
    demo_cost_comparison()

    print("=" * 60)
    print("KEY TAKEAWAYS:")
    print("  1. Every LLM call has input + output tokens -> costs money")
    print("  2. Tool agents make MULTIPLE LLM calls per query")
    print("  3. Use CostTracker to monitor and alert on spending")
    print("  4. Groq/Llama and Gemini Flash are affordable for learning")
    print("  5. Set weekly budgets to avoid bill shock")
    print("  6. Extract token_usage from response.response_metadata")
    print("=" * 60)
