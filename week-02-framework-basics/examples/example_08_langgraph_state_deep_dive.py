"""
Example 8: LangGraph State Management Deep Dive
==================================================
Example 01 introduced state basics. This goes deeper into how
LangGraph manages state across complex workflows:

  1. Multiple state fields with different update strategies
  2. State as the agent's "working memory"
  3. Reading previous state to make decisions
  4. Accumulating data across multiple nodes
  5. Using state for inter-node communication

Key insight: In LangGraph, STATE is how nodes talk to each other.
There are no function arguments passed between nodes -- everything
goes through the shared state dictionary.

Run: python week-02-framework-basics/examples/example_08_langgraph_state_deep_dive.py
"""

import os
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, Annotated, Optional, List
from langgraph.graph import add_messages


# ================================================================
# PART 1: Rich State -- Multiple fields for complex workflows
# ================================================================
# Real agents need more than just messages. State can hold any
# data your nodes need: counters, flags, collected results, etc.

class ResearchState(TypedDict):
    # Conversation history (reducer: appends new messages)
    messages: Annotated[list, add_messages]

    # Research metadata (plain fields: each write REPLACES)
    topic: str
    facts: List[str]           # Collected facts
    summary: str               # Final summary
    confidence: float          # How confident are we (0.0 - 1.0)
    iteration: int             # Which pass are we on
    needs_more_research: bool  # Should we do another pass?


def demo_rich_state():
    """Show a multi-node workflow using rich state."""
    print("=" * 60)
    print("PART 1: Rich State (multiple fields, multi-node workflow)")
    print("=" * 60)

    # Set up LLM
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    else:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    # -- Node 1: Collect facts about the topic --
    def collect_facts(state: ResearchState) -> dict:
        """Ask the LLM for facts and store them in state."""
        iteration = state.get("iteration", 0) + 1
        existing_facts = state.get("facts", [])

        prompt = f"List 3 key facts about: {state['topic']}"
        if existing_facts:
            prompt += f"\nYou already know: {existing_facts}. Find NEW facts."

        response = llm.invoke(prompt)
        new_facts = [line.strip() for line in response.content.split("\n") if line.strip()]

        print(f"  [collect_facts] Iteration {iteration}: found {len(new_facts)} facts")
        return {
            "facts": existing_facts + new_facts,  # Accumulate facts
            "iteration": iteration,
        }

    # -- Node 2: Assess confidence --
    def assess_confidence(state: ResearchState) -> dict:
        """Decide if we have enough facts or need more research."""
        num_facts = len(state.get("facts", []))
        iteration = state.get("iteration", 0)

        # Simple heuristic: more facts = more confidence
        confidence = min(num_facts / 6.0, 1.0)  # 6 facts = full confidence
        needs_more = confidence < 0.8 and iteration < 3  # Max 3 passes

        print(f"  [assess] {num_facts} facts, confidence={confidence:.1f}, "
              f"needs_more={needs_more}")
        return {
            "confidence": confidence,
            "needs_more_research": needs_more,
        }

    # -- Node 3: Summarize findings --
    def summarize(state: ResearchState) -> dict:
        """Create a summary from collected facts."""
        facts_text = "\n".join(f"- {f}" for f in state["facts"] if f)
        prompt = (f"Summarize these facts about '{state['topic']}' "
                  f"in 2-3 sentences:\n{facts_text}")
        response = llm.invoke(prompt)
        print(f"  [summarize] Generated summary")
        return {"summary": response.content}

    # -- Routing: need more research? --
    def should_continue_research(state: ResearchState) -> str:
        if state.get("needs_more_research", False):
            return "collect"  # Loop back for more facts
        return "summarize"    # We have enough

    # -- Build the graph --
    graph = StateGraph(ResearchState)
    graph.add_node("collect", collect_facts)
    graph.add_node("assess", assess_confidence)
    graph.add_node("summarize", summarize)

    graph.set_entry_point("collect")
    graph.add_edge("collect", "assess")
    graph.add_conditional_edges(
        "assess", should_continue_research,
        {"collect": "collect", "summarize": "summarize"},
    )
    graph.add_edge("summarize", END)

    app = graph.compile()

    # Run it
    print(f"\nTopic: 'Renewable Energy'\n")
    result = app.invoke({
        "messages": [],
        "topic": "Renewable Energy",
        "facts": [],
        "summary": "",
        "confidence": 0.0,
        "iteration": 0,
        "needs_more_research": True,
    })

    print(f"\nResults:")
    print(f"  Iterations:  {result['iteration']}")
    print(f"  Facts found: {len(result['facts'])}")
    print(f"  Confidence:  {result['confidence']:.1f}")
    print(f"  Summary:     {result['summary'][:200]}...")
    print()


# ================================================================
# PART 2: State as Inter-Node Communication
# ================================================================

def demo_state_communication():
    """Show how nodes communicate ONLY through state."""
    print("=" * 60)
    print("PART 2: State as Inter-Node Communication")
    print("=" * 60)

    class PipelineState(TypedDict):
        raw_input: str
        cleaned: str
        word_list: List[str]
        stats: dict
        final_report: str

    # Each node reads what previous nodes wrote to state

    def clean_node(state: PipelineState) -> dict:
        """Step 1: Clean the input text."""
        cleaned = state["raw_input"].strip().lower()
        print(f"  [clean] '{state['raw_input'][:30]}' -> '{cleaned[:30]}'")
        return {"cleaned": cleaned}

    def split_node(state: PipelineState) -> dict:
        """Step 2: Split into words (reads 'cleaned' from step 1)."""
        words = state["cleaned"].split()
        print(f"  [split] Found {len(words)} words")
        return {"word_list": words}

    def stats_node(state: PipelineState) -> dict:
        """Step 3: Compute stats (reads 'word_list' from step 2)."""
        words = state["word_list"]
        stats = {
            "word_count": len(words),
            "unique_words": len(set(words)),
            "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
        }
        print(f"  [stats] {stats}")
        return {"stats": stats}

    def report_node(state: PipelineState) -> dict:
        """Step 4: Generate report (reads 'stats' from step 3)."""
        s = state["stats"]
        report = (f"Text Analysis Report:\n"
                  f"  Words: {s['word_count']}\n"
                  f"  Unique: {s['unique_words']}\n"
                  f"  Avg length: {s['avg_word_length']:.1f} chars")
        print(f"  [report] Generated")
        return {"final_report": report}

    graph = StateGraph(PipelineState)
    graph.add_node("clean", clean_node)
    graph.add_node("split", split_node)
    graph.add_node("stats", stats_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("clean")
    graph.add_edge("clean", "split")
    graph.add_edge("split", "stats")
    graph.add_edge("stats", "report")
    graph.add_edge("report", END)

    app = graph.compile()

    text = "  LangGraph State Management is POWERFUL and Flexible  "
    print(f"\nInput: '{text}'\n")
    result = app.invoke({
        "raw_input": text,
        "cleaned": "",
        "word_list": [],
        "stats": {},
        "final_report": "",
    })

    print(f"\n{result['final_report']}")
    print(f"\nKey insight: Each node only reads what it needs from state")
    print(f"  clean -> writes 'cleaned'")
    print(f"  split -> reads 'cleaned', writes 'word_list'")
    print(f"  stats -> reads 'word_list', writes 'stats'")
    print(f"  report -> reads 'stats', writes 'final_report'")
    print()


# ================================================================
# Run all demos
# ================================================================

if __name__ == "__main__":
    print("\nLangGraph State Management Deep Dive")
    print("=" * 60)

    demo_rich_state()
    demo_state_communication()

    print("=" * 60)
    print("KEY TAKEAWAYS:")
    print("  1. State can hold any data: strings, lists, dicts, flags")
    print("  2. Plain fields are REPLACED; Annotated fields use reducers")
    print("  3. Nodes communicate ONLY through shared state")
    print("  4. Use state flags (like needs_more_research) for routing")
    print("  5. Accumulate data across iterations using lists")
    print("=" * 60)
