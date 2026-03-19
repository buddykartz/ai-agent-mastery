"""
Example 9: ADK Workflow Agents -- Sequential Pipelines
========================================================
ADK supports multi-agent workflows where agents execute in sequence.
A SequentialAgent runs a list of sub-agents one after another,
passing context between them through the session.

This is like a LangGraph linear graph (node1 -> node2 -> node3)
but configured declaratively instead of built with code.

Use cases:
  - Research pipeline: gather -> analyze -> summarize
  - Content pipeline: draft -> review -> polish
  - Data pipeline: extract -> transform -> validate

Requires: GOOGLE_API_KEY in your environment.

Run: python week-02-framework-basics/examples/example_09_adk_sequential_agent.py
"""

import asyncio
import logging
import os
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

logging.getLogger("google_genai.types").setLevel(logging.ERROR)

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


# ================================================================
# PART 1: Simple Sequential Pipeline
# ================================================================
# Three agents run in order: researcher -> analyst -> writer.
# Each agent picks up where the previous one left off.

model = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")

# -- Agent 1: Researcher (gathers raw information) --
researcher = LlmAgent(
    name="researcher",
    model=model,
    instruction="""You are a research assistant. When given a topic,
    list 5 key facts about it. Be concise -- one sentence per fact.
    Format as a numbered list.""",
    description="Gathers key facts about a topic.",
)

# -- Agent 2: Analyst (interprets the facts) --
analyst = LlmAgent(
    name="analyst",
    model=model,
    instruction="""You are a data analyst. Look at the facts provided
    in the conversation and identify:
    1. The most important finding
    2. A surprising or lesser-known detail
    3. A potential area for further investigation
    Be concise -- 2-3 sentences total.""",
    description="Analyzes facts and identifies key insights.",
)

# -- Agent 3: Writer (produces final output) --
writer = LlmAgent(
    name="writer",
    model=model,
    instruction="""You are a technical writer. Based on the research
    and analysis in the conversation, write a brief summary paragraph
    (3-4 sentences) suitable for a newsletter. Make it engaging and
    informative. Start with the most important point.""",
    description="Writes a polished summary from research and analysis.",
)

# -- SequentialAgent: runs them in order --
# The session carries the conversation across all three agents.
# Each agent sees what the previous agents wrote.
pipeline = SequentialAgent(
    name="research_pipeline",
    sub_agents=[researcher, analyst, writer],
    description="A 3-stage research pipeline: gather -> analyze -> write.",
)


async def demo_sequential_pipeline():
    """Run the sequential research pipeline."""
    print("=" * 60)
    print("PART 1: Sequential Pipeline (researcher -> analyst -> writer)")
    print("=" * 60)

    session_service = InMemorySessionService()
    runner = Runner(
        agent=pipeline,
        app_name="pipeline_app",
        session_service=session_service,
    )

    session = await session_service.create_session(
        app_name="pipeline_app", user_id="user1"
    )

    topic = "The impact of AI on healthcare"
    print(f"\nTopic: {topic}\n")
    print("Running pipeline: researcher -> analyst -> writer...\n")

    async for event in runner.run_async(
        user_id="user1",
        session_id=session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=f"Research this topic: {topic}")],
        ),
    ):
        # Show which agent is responding
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    agent_name = getattr(event, "author", "unknown")
                    if event.is_final_response():
                        print(f"[FINAL - {agent_name}]")
                        print(f"{part.text}\n")
                    else:
                        print(f"[{agent_name}] {part.text[:100]}...")
                        print()

    print("Pipeline complete!")
    print()


# ================================================================
# PART 2: Sequential Agent with Tools
# ================================================================

async def demo_sequential_with_tools():
    """Sequential pipeline where one agent has tools."""
    print("=" * 60)
    print("PART 2: Sequential Agent with Tools")
    print("=" * 60)

    # Tool for the data agent
    def count_words(text: str) -> str:
        """Count words in a text. Returns the count as a string."""
        count = len(text.split())
        return f"The text has {count} words."

    # Agent 1: Generate content
    generator = LlmAgent(
        name="generator",
        model=model,
        instruction="Write a 2-sentence explanation of what LangGraph is.",
    )

    # Agent 2: Analyze with tools
    analyzer = LlmAgent(
        name="analyzer",
        model=model,
        instruction="""Analyze the text from the previous response.
        Use the count_words tool to count its words.
        Then comment on whether the length is appropriate.""",
        tools=[count_words],
    )

    pipeline = SequentialAgent(
        name="generate_and_analyze",
        sub_agents=[generator, analyzer],
    )

    session_service = InMemorySessionService()
    runner = Runner(
        agent=pipeline,
        app_name="tools_pipeline",
        session_service=session_service,
    )

    session = await session_service.create_session(
        app_name="tools_pipeline", user_id="user1"
    )

    print("\nRunning: generator -> analyzer (with word count tool)...\n")

    async for event in runner.run_async(
        user_id="user1",
        session_id=session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="Generate and analyze")],
        ),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    print(f"Final output: {part.text}")
    print()


# ================================================================
# Run all demos
# ================================================================

if __name__ == "__main__":
    print("\nADK Sequential Agents -- Multi-Agent Pipelines")
    print("=" * 60)
    print(f"Model: {model}\n")

    async def run_all():
        await demo_sequential_pipeline()
        await demo_sequential_with_tools()

    asyncio.run(run_all())

    print("=" * 60)
    print("KEY TAKEAWAYS:")
    print("  1. SequentialAgent runs sub-agents in order")
    print("  2. Each agent sees the full conversation from previous agents")
    print("  3. No explicit state wiring needed -- session carries context")
    print("  4. Sub-agents can have their own tools")
    print("  5. Compare: LangGraph needs explicit edges; ADK just lists agents")
    print("=" * 60)
