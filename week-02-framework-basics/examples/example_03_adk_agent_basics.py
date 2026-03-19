"""
Example 3: ADK Agent Basics — Google's Agent Development Kit
==============================================================
Google ADK takes a different approach from LangGraph. Instead of
building a graph manually, you configure an agent declaratively and
ADK handles the execution loop for you.

Key Concepts:
  - LlmAgent: the main agent class (name, model, instruction, tools)
  - Runner: executes the agent and manages the event loop
  - InMemorySessionService: stores conversation history per session
  - Events: the streaming response mechanism (tool calls, text, etc.)
  - Sessions: isolated conversation contexts (like separate chat windows)

ADK uses Gemini models by default (requires GOOGLE_API_KEY).

Run: python week-02-framework-basics/examples/example_03_adk_agent_basics.py
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


# ══════════════════════════════════════════════════════════════
# PART 1: Minimal Agent — The simplest possible ADK agent
# ══════════════════════════════════════════════════════════════

# ── Step 1: Create an agent ─────────────────────────────────
# LlmAgent needs just 3 things: name, model, and instruction.
# The instruction is like a system prompt — it tells the agent
# how to behave.

simple_agent = LlmAgent(
    name="tutor",                                         # Unique agent name
    model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),  # Gemini model
    instruction="You are a helpful AI tutor. Give clear, concise explanations. "
                "Always end with a follow-up question to check understanding.",
)

# ── Step 2: Set up Runner and SessionService ────────────────
# Runner = the execution engine (sends messages, handles responses)
# SessionService = stores conversation memory
# Think of Runner as the "engine" and SessionService as the "memory bank"

session_service = InMemorySessionService()
runner = Runner(
    agent=simple_agent,
    app_name="tutor_app",          # Groups sessions under one app
    session_service=session_service,
)


# ── Step 3: Send a message and get a response ──────────────
# ADK uses an async event stream. We iterate over events and look
# for the final response.

async def ask_simple(query: str) -> str:
    """Send a single query to the agent and return the response."""
    # Create a session (a conversation context)
    session = await session_service.create_session(
        app_name="tutor_app",
        user_id="student",
    )

    # Send the message and collect the response
    response_text = ""
    async for event in runner.run_async(
        user_id="student",
        session_id=session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=query)],
        ),
    ):
        # is_final_response() means the agent is done thinking
        if event.is_final_response():
            response_text = event.content.parts[0].text

    return response_text


async def demo_minimal_agent():
    """Demonstrate the simplest ADK agent."""
    print("=" * 60)
    print("PART 1: Minimal ADK Agent")
    print("=" * 60)

    response = await ask_simple("What is an AI agent in 2 sentences?")
    print(f"\nQuery: What is an AI agent in 2 sentences?")
    print(f"Agent: {response}\n")


# ══════════════════════════════════════════════════════════════
# PART 2: Agent Configuration Options
# ══════════════════════════════════════════════════════════════

async def demo_configured_agent():
    """Show various configuration options for ADK agents."""
    print("=" * 60)
    print("PART 2: Agent Configuration Options")
    print("=" * 60)

    # You can configure the agent with additional parameters
    configured_agent = LlmAgent(
        name="creative_writer",
        model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),

        # 'instruction' is the system prompt — the agent's personality
        instruction="""You are a creative writing assistant.
        - Write in a vivid, engaging style
        - Keep responses under 100 words
        - Use metaphors and sensory details""",

        # 'description' helps in multi-agent systems (Week 4+)
        # Other agents read this to understand what this agent does
        description="A creative writer that produces vivid, concise prose.",

        # 'generate_content_config' controls generation parameters
        generate_content_config=types.GenerateContentConfig(
            temperature=0.9,          # Higher = more creative (0.0-2.0)
            max_output_tokens=200,    # Limit response length
        ),
    )

    # Set up a new runner for this agent
    config_session_service = InMemorySessionService()
    config_runner = Runner(
        agent=configured_agent,
        app_name="writer_app",
        session_service=config_session_service,
    )

    session = await config_session_service.create_session(
        app_name="writer_app", user_id="user1"
    )

    query = "Describe a sunset over the ocean in 2 sentences."
    print(f"\nQuery: {query}")

    async for event in config_runner.run_async(
        user_id="user1",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=query)]),
    ):
        if event.is_final_response():
            print(f"Agent: {event.content.parts[0].text}")

    print(f"\nConfiguration used:")
    print(f"  temperature:      0.9 (high creativity)")
    print(f"  max_output_tokens: 200")
    print()


# ══════════════════════════════════════════════════════════════
# PART 3: Multi-Turn Conversations (Session Memory)
# ══════════════════════════════════════════════════════════════

async def demo_multi_turn():
    """Show how ADK maintains context across conversation turns."""
    print("=" * 60)
    print("PART 3: Multi-Turn Conversation (Automatic Memory)")
    print("=" * 60)

    # Create a fresh agent and session service
    memory_agent = LlmAgent(
        name="memory_demo",
        model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
        instruction="You are a helpful assistant. Remember everything the user "
                    "tells you. Use their name if they share it.",
    )

    mem_session_service = InMemorySessionService()
    mem_runner = Runner(
        agent=memory_agent,
        app_name="memory_app",
        session_service=mem_session_service,
    )

    # Create ONE session — all turns within this session share memory
    session = await mem_session_service.create_session(
        app_name="memory_app", user_id="user1"
    )

    # Send 3 messages in the SAME session
    conversations = [
        "Hi! My name is Alex and I'm learning about AI agents.",
        "What are the two frameworks we're studying this week?",
        "What's my name? And what am I learning about?",
    ]

    for turn_num, message in enumerate(conversations, 1):
        print(f"\n  Turn {turn_num}")
        print(f"  User: {message}")

        async for event in mem_runner.run_async(
            user_id="user1",
            session_id=session.id,  # Same session = same memory!
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=message)],
            ),
        ):
            if event.is_final_response():
                print(f"  Agent: {event.content.parts[0].text}")

    # KEY INSIGHT: ADK automatically maintained the full conversation
    # history within this session. No manual state management needed!
    # Compare this with LangGraph where you manage messages explicitly.
    print(f"\n  [All 3 turns used session_id={session.id}]")
    print(f"  [ADK automatically kept the full conversation history]")
    print()


# ══════════════════════════════════════════════════════════════
# PART 4: Session Isolation — Different sessions, different memory
# ══════════════════════════════════════════════════════════════

async def demo_session_isolation():
    """Show that different sessions do NOT share memory."""
    print("=" * 60)
    print("PART 4: Session Isolation")
    print("=" * 60)

    iso_agent = LlmAgent(
        name="isolation_demo",
        model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
        instruction="You are a helpful assistant. If the user told you their "
                    "name, use it. If not, say you don't know their name.",
    )

    iso_session_service = InMemorySessionService()
    iso_runner = Runner(
        agent=iso_agent,
        app_name="isolation_app",
        session_service=iso_session_service,
    )

    # Session A: Tell the agent our name
    session_a = await iso_session_service.create_session(
        app_name="isolation_app", user_id="user1"
    )
    print(f"\n  Session A (id={session_a.id[:8]}...):")
    print(f"  User: My name is Bob.")
    async for event in iso_runner.run_async(
        user_id="user1",
        session_id=session_a.id,
        new_message=types.Content(role="user", parts=[types.Part(text="My name is Bob.")]),
    ):
        if event.is_final_response():
            print(f"  Agent: {event.content.parts[0].text}")

    # Session B: Ask for our name (different session = no memory of Bob)
    session_b = await iso_session_service.create_session(
        app_name="isolation_app", user_id="user1"
    )
    print(f"\n  Session B (id={session_b.id[:8]}...):")
    print(f"  User: What is my name?")
    async for event in iso_runner.run_async(
        user_id="user1",
        session_id=session_b.id,
        new_message=types.Content(role="user", parts=[types.Part(text="What is my name?")]),
    ):
        if event.is_final_response():
            print(f"  Agent: {event.content.parts[0].text}")

    print(f"\n  [Session A and B are isolated -- B doesn't know about Bob!]")
    print(f"  [This is like opening two separate chat windows]")
    print()


# ══════════════════════════════════════════════════════════════
# Run all demos
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\nADK Agent Basics -- Google's Agent Development Kit")
    print("=" * 60)
    print(f"Model: {os.getenv('GOOGLE_MODEL', 'gemini-2.0-flash')}\n")

    async def run_all():
        await demo_minimal_agent()
        await demo_configured_agent()
        await demo_multi_turn()
        await demo_session_isolation()

    asyncio.run(run_all())

    print("=" * 60)
    print("KEY TAKEAWAYS:")
    print("  1. LlmAgent = declarative config (name, model, instruction)")
    print("  2. Runner = execution engine, SessionService = memory store")
    print("  3. Sessions maintain conversation history automatically")
    print("  4. Different sessions are isolated (no shared memory)")
    print("  5. No graph building needed -- ADK handles the loop for you")
    print("  6. Compare: LangGraph = explicit control, ADK = convenience")
    print("=" * 60)
