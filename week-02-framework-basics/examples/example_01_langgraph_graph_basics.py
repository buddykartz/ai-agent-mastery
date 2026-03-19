"""
Example 1: LangGraph Graph Basics — Understanding Graphs Without LLMs
=======================================================================
Before using LLMs, let's understand the core abstraction of LangGraph:
the GRAPH. A graph has nodes (functions), edges (connections), and state
(shared data). This example uses NO LLM calls — just pure Python
functions — so you can see how data flows through a graph.

Key Concepts:
  - StateGraph: the main class that holds your graph
  - Nodes: Python functions that read/write shared state
  - Edges: connections between nodes (linear or conditional)
  - State: a TypedDict that all nodes share
  - Reducers: how state fields get updated (e.g., add_messages appends)

Run: python week-02-framework-basics/examples/example_01_langgraph_graph_basics.py
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langgraph.graph import add_messages


# ══════════════════════════════════════════════════════════════
# PART 1: Linear Graph — Nodes run one after another
# ══════════════════════════════════════════════════════════════

# ── Step 1: Define the state ────────────────────────────────
# State is a TypedDict — a dictionary with specific keys and types.
# Every node in the graph can READ and WRITE to this shared state.

class LinearState(TypedDict):
    # 'input_text' is a plain field — each write REPLACES the old value
    input_text: str
    # 'processed_text' stores the result after processing
    processed_text: str
    # 'step_log' tracks which nodes ran (plain list — replaced on write)
    step_log: list


# ── Step 2: Define node functions ───────────────────────────
# Each node is a function that takes 'state' and returns a dict
# with the fields it wants to update. It does NOT need to return
# all fields — only the ones it changes.

def intake_node(state: LinearState) -> dict:
    """First node: receives input and logs that we started."""
    print("  [intake_node] Received input:", state["input_text"])

    # Return only the fields we want to update
    return {
        "step_log": state.get("step_log", []) + ["intake"],
    }


def process_node(state: LinearState) -> dict:
    """Second node: transforms the input text."""
    text = state["input_text"]

    # Simple processing: uppercase and add word count
    processed = f"{text.upper()} (word count: {len(text.split())})"
    print("  [process_node] Processed:", processed)

    return {
        "processed_text": processed,
        "step_log": state.get("step_log", []) + ["process"],
    }


def output_node(state: LinearState) -> dict:
    """Third node: finalizes the result."""
    print("  [output_node] Final result:", state["processed_text"])

    return {
        "step_log": state.get("step_log", []) + ["output"],
    }


# ── Step 3: Build the linear graph ─────────────────────────
# This creates: intake → process → output → END

def build_linear_graph():
    """Build a simple 3-node linear graph."""
    graph = StateGraph(LinearState)

    # Add nodes — each gets a name (string) and a function
    graph.add_node("intake", intake_node)
    graph.add_node("process", process_node)
    graph.add_node("output", output_node)

    # Set the entry point — which node runs first
    graph.set_entry_point("intake")

    # Add edges — linear chain: intake → process → output → END
    graph.add_edge("intake", "process")
    graph.add_edge("process", "output")
    graph.add_edge("output", END)

    # Compile turns the graph definition into a runnable app
    return graph.compile()


def demo_linear_graph():
    """Run the linear graph and show state at each step."""
    print("=" * 60)
    print("PART 1: Linear Graph (intake -> process -> output)")
    print("=" * 60)

    app = build_linear_graph()

    # Invoke the graph with initial state
    initial_state = {
        "input_text": "LangGraph makes agent workflows visual",
        "processed_text": "",
        "step_log": [],
    }

    print(f"\nInitial state: {initial_state}\n")
    result = app.invoke(initial_state)

    print(f"\nFinal state:")
    print(f"  input_text:    {result['input_text']}")
    print(f"  processed_text:{result['processed_text']}")
    print(f"  step_log:      {result['step_log']}")
    print()


# ══════════════════════════════════════════════════════════════
# PART 2: Conditional Graph — Edges that choose where to go
# ══════════════════════════════════════════════════════════════

# ── Step 4: State with a counter for conditional logic ──────

class ConditionalState(TypedDict):
    value: int           # A number we'll keep doubling
    step_count: int      # How many times we've processed
    max_steps: int       # When to stop
    history: list        # Track the value at each step


# ── Step 5: Nodes for the conditional graph ─────────────────

def init_node(state: ConditionalState) -> dict:
    """Initialize the processing."""
    print(f"  [init_node] Starting with value={state['value']}")
    return {
        "history": [state["value"]],
        "step_count": 0,
    }


def double_node(state: ConditionalState) -> dict:
    """Double the current value."""
    new_value = state["value"] * 2
    new_count = state["step_count"] + 1
    print(f"  [double_node] Step {new_count}: {state['value']} -> {new_value}")
    return {
        "value": new_value,
        "step_count": new_count,
        "history": state.get("history", []) + [new_value],
    }


def done_node(state: ConditionalState) -> dict:
    """Final node — just logs completion."""
    print(f"  [done_node] Finished after {state['step_count']} steps. Final value: {state['value']}")
    return {}


# ── Step 6: The conditional routing function ────────────────
# This function decides which node to go to next.
# It MUST return a string matching one of the edge targets.

def should_continue(state: ConditionalState) -> str:
    """Decide whether to keep doubling or stop.

    Returns:
        'double' to loop back and double again
        'done'   to stop processing
    """
    if state["step_count"] < state["max_steps"]:
        return "double"  # Keep going
    else:
        return "done"    # We've hit our limit


# ── Step 7: Build the conditional graph ─────────────────────
# This creates:  init → double ⟲ (loop while step_count < max_steps) → done → END

def build_conditional_graph():
    """Build a graph with a conditional loop."""
    graph = StateGraph(ConditionalState)

    graph.add_node("init", init_node)
    graph.add_node("double", double_node)
    graph.add_node("done", done_node)

    graph.set_entry_point("init")

    # After init, always go to double
    graph.add_edge("init", "double")

    # After double, CHECK the condition:
    # - If should_continue returns "double" → go back to double (LOOP!)
    # - If should_continue returns "done" → go to done node
    graph.add_conditional_edges(
        "double",              # Source node
        should_continue,       # Function that decides
        {                      # Map: return value → target node
            "double": "double",  # Loop back
            "done": "done",      # Move forward
        },
    )

    graph.add_edge("done", END)

    return graph.compile()


def demo_conditional_graph():
    """Run the conditional graph that loops until a limit."""
    print("=" * 60)
    print("PART 2: Conditional Graph (loop until max_steps)")
    print("=" * 60)

    app = build_conditional_graph()

    initial_state = {
        "value": 1,
        "step_count": 0,
        "max_steps": 4,   # Will double 4 times: 1 → 2 → 4 → 8 → 16
        "history": [],
    }

    print(f"\nStarting: value={initial_state['value']}, max_steps={initial_state['max_steps']}\n")
    result = app.invoke(initial_state)

    print(f"\nFinal state:")
    print(f"  value:      {result['value']}")
    print(f"  step_count: {result['step_count']}")
    print(f"  history:    {result['history']}")
    print()


# ══════════════════════════════════════════════════════════════
# PART 3: Branching Graph — Different paths based on input
# ══════════════════════════════════════════════════════════════

class BranchState(TypedDict):
    text: str
    category: str     # "short", "medium", or "long"
    result: str


def classify_node(state: BranchState) -> dict:
    """Classify text by length into categories."""
    word_count = len(state["text"].split())
    if word_count <= 3:
        category = "short"
    elif word_count <= 10:
        category = "medium"
    else:
        category = "long"
    print(f"  [classify] '{state['text']}' -> {category} ({word_count} words)")
    return {"category": category}


def handle_short(state: BranchState) -> dict:
    """Handle short texts — expand them."""
    result = f"[SHORT] Your text '{state['text']}' is brief. Consider adding more detail."
    print(f"  [handle_short] {result}")
    return {"result": result}


def handle_medium(state: BranchState) -> dict:
    """Handle medium texts — summarize them."""
    result = f"[MEDIUM] Your text has a good length. Summary: '{state['text'][:50]}...'"
    print(f"  [handle_medium] {result}")
    return {"result": result}


def handle_long(state: BranchState) -> dict:
    """Handle long texts — warn about length."""
    word_count = len(state["text"].split())
    result = f"[LONG] Your text has {word_count} words. Consider breaking it into sections."
    print(f"  [handle_long] {result}")
    return {"result": result}


def route_by_category(state: BranchState) -> str:
    """Route to the appropriate handler based on category."""
    return state["category"]  # Returns "short", "medium", or "long"


def build_branching_graph():
    """Build a graph where different inputs take different paths."""
    graph = StateGraph(BranchState)

    graph.add_node("classify", classify_node)
    graph.add_node("short", handle_short)
    graph.add_node("medium", handle_medium)
    graph.add_node("long", handle_long)

    graph.set_entry_point("classify")

    # After classify, BRANCH to one of three handlers
    graph.add_conditional_edges(
        "classify",
        route_by_category,
        {
            "short": "short",
            "medium": "medium",
            "long": "long",
        },
    )

    # All three handlers converge at END
    graph.add_edge("short", END)
    graph.add_edge("medium", END)
    graph.add_edge("long", END)

    return graph.compile()


def demo_branching_graph():
    """Run the branching graph with different inputs."""
    print("=" * 60)
    print("PART 3: Branching Graph (route by text length)")
    print("=" * 60)

    app = build_branching_graph()

    test_inputs = [
        "Hello world",
        "LangGraph is great for building agent workflows easily",
        "This is a very long text that has many words and should be classified as long because it exceeds the threshold of ten words by quite a lot",
    ]

    for text in test_inputs:
        print(f"\nInput: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        result = app.invoke({"text": text, "category": "", "result": ""})
        print(f"  -> Result: {result['result']}")
    print()


# ══════════════════════════════════════════════════════════════
# PART 4: Understanding State Reducers (add_messages)
# ══════════════════════════════════════════════════════════════

class ReducerState(TypedDict):
    # WITHOUT a reducer: each write REPLACES the entire list
    plain_list: list

    # WITH a reducer (add_messages): each write APPENDS to the list
    # This is how LangGraph handles conversation history!
    reducer_list: Annotated[list, add_messages]


def append_demo_node_a(state: ReducerState) -> dict:
    """First node adds some items."""
    from langchain_core.messages import HumanMessage
    return {
        "plain_list": ["a", "b"],            # Replaces whatever was there
        "reducer_list": [HumanMessage(content="Hello")],  # Appends via add_messages
    }


def append_demo_node_b(state: ReducerState) -> dict:
    """Second node adds more items — see the difference!"""
    from langchain_core.messages import HumanMessage
    return {
        "plain_list": ["c", "d"],            # REPLACES ["a", "b"] with ["c", "d"]
        "reducer_list": [HumanMessage(content="World")],  # APPENDS — list now has both messages
    }


def demo_reducers():
    """Show the difference between plain fields and reducer fields."""
    print("=" * 60)
    print("PART 4: State Reducers -- Replace vs Append")
    print("=" * 60)

    graph = StateGraph(ReducerState)
    graph.add_node("node_a", append_demo_node_a)
    graph.add_node("node_b", append_demo_node_b)
    graph.set_entry_point("node_a")
    graph.add_edge("node_a", "node_b")
    graph.add_edge("node_b", END)

    app = graph.compile()
    result = app.invoke({"plain_list": [], "reducer_list": []})

    print(f"\n  plain_list (no reducer):  {result['plain_list']}")
    print(f"    -> Node A wrote ['a','b'], Node B wrote ['c','d']")
    print(f"    -> Final value is ['c','d'] because Node B REPLACED Node A's value")

    print(f"\n  reducer_list (add_messages): {len(result['reducer_list'])} messages")
    for msg in result["reducer_list"]:
        print(f"    -> {msg.content}")
    print(f"    -> Both messages are kept because add_messages APPENDS")
    print()


# ══════════════════════════════════════════════════════════════
# Run all demos
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\nLangGraph Graph Basics -- No LLM, Just Graphs!")
    print("=" * 60)
    print("This example teaches graph mechanics using pure Python.\n")

    demo_linear_graph()
    demo_conditional_graph()
    demo_branching_graph()
    demo_reducers()

    print("=" * 60)
    print("KEY TAKEAWAYS:")
    print("  1. Graphs have nodes (functions) and edges (connections)")
    print("  2. State is a TypedDict shared by all nodes")
    print("  3. Conditional edges let you route based on state")
    print("  4. Reducers (like add_messages) control how state updates")
    print("  5. These same patterns power LLM agents -- next example!")
    print("=" * 60)
