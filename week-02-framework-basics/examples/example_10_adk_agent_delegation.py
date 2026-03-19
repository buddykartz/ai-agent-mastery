"""
Example 10: ADK AgentTool Delegation
=======================================
In ADK, an agent can use OTHER AGENTS as tools. This is called
"agent delegation" via AgentTool. The parent agent decides when
to delegate a task to a specialized sub-agent.

Think of it like a manager delegating tasks:
  - Parent agent: "I need weather data" -> delegates to weather_agent
  - Parent agent: "I need a calculation" -> delegates to math_agent
  - Parent agent: combines results and responds

This is different from SequentialAgent (example 09):
  - SequentialAgent: agents ALWAYS run in fixed order
  - AgentTool: parent DECIDES when to call sub-agents (like tools)

Requires: GOOGLE_API_KEY in your environment.

Run: python week-02-framework-basics/examples/example_10_adk_agent_delegation.py
"""

import asyncio
import logging
import os
import requests
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

logging.getLogger("google_genai.types").setLevel(logging.ERROR)

from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


model = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")


# ================================================================
# PART 1: Agent Delegation Basics
# ================================================================

# -- Specialist Agent 1: Math Expert --
def calculate(expression: str) -> str:
    """Evaluate a math expression. Example: '15 * 7', '2 ** 10'"""
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters"
        return f"{expression} = {eval(expression)}"
    except Exception as e:
        return f"Error: {e}"

math_agent = LlmAgent(
    name="math_expert",
    model=model,
    instruction="You are a math expert. Use the calculate tool to solve math problems. "
                "Show your work step by step.",
    tools=[calculate],
    description="Solves math problems with step-by-step explanations.",
)

# -- Specialist Agent 2: Weather Expert --
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1}, timeout=10,
        ).json()
        if geo.get("results"):
            loc = geo["results"][0]
            w = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={"latitude": loc["latitude"], "longitude": loc["longitude"],
                         "current_weather": True},
                timeout=10,
            ).json()
            cw = w["current_weather"]
            return f"Weather in {loc['name']}: {cw['temperature']}C, wind {cw['windspeed']} km/h"
    except Exception:
        pass
    return f"Could not fetch weather for {city}"

weather_agent = LlmAgent(
    name="weather_expert",
    model=model,
    instruction="You are a weather expert. Use the get_weather tool to check conditions. "
                "Give a brief, friendly weather report.",
    tools=[get_weather],
    description="Checks weather conditions for any city worldwide.",
)

# -- Parent Agent: Delegates to specialists --
# The parent uses AgentTool to wrap each specialist agent.
# The parent's LLM sees these as callable tools and decides
# which specialist to invoke based on the user's question.

parent_agent = LlmAgent(
    name="coordinator",
    model=model,
    instruction="""You are a helpful coordinator agent.
    You have access to specialist agents:
    - math_expert: for any math calculations
    - weather_expert: for weather information

    Delegate tasks to the appropriate specialist.
    Combine their answers into a clear response for the user.""",
    tools=[
        AgentTool(agent=math_agent),
        AgentTool(agent=weather_agent),
    ],
    description="Coordinates between math and weather specialists.",
)


async def demo_agent_delegation():
    """Show the parent agent delegating to specialists."""
    print("=" * 60)
    print("PART 1: Agent Delegation (parent delegates to specialists)")
    print("=" * 60)

    session_service = InMemorySessionService()
    runner = Runner(
        agent=parent_agent,
        app_name="delegation_app",
        session_service=session_service,
    )

    queries = [
        "What is 25 * 17?",
        "What's the weather in Tokyo?",
        "Calculate 100 / 7 and also tell me the weather in London",
    ]

    for query in queries:
        session = await session_service.create_session(
            app_name="delegation_app", user_id="user1"
        )

        print(f"\nQuery: {query}")
        async for event in runner.run_async(
            user_id="user1",
            session_id=session.id,
            new_message=types.Content(
                role="user", parts=[types.Part(text=query)]
            ),
        ):
            if event.is_final_response() and event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        print(f"Coordinator: {part.text}")
    print()


# ================================================================
# PART 2: Comparing Delegation Patterns
# ================================================================

async def demo_comparison():
    """Explain the difference between tools, sequential, and delegation."""
    print("=" * 60)
    print("PART 2: When to Use Each Pattern")
    print("=" * 60)

    print("""
  Pattern         | How it works               | When to use
  --------------- | -------------------------- | -------------------------
  Plain tools     | Functions the LLM calls    | Simple, stateless operations
  SequentialAgent | Agents run in fixed order  | Pipelines (A then B then C)
  AgentTool       | Parent chooses which agent | Dynamic delegation by task

  Plain tools (example 04):
    agent -> calculate() -> agent -> respond
    Best for: simple functions, API calls

  SequentialAgent (example 09):
    researcher -> analyst -> writer (always this order)
    Best for: fixed pipelines, content generation

  AgentTool (this example):
    coordinator -> [math_expert OR weather_expert] -> coordinator
    Best for: dynamic routing, specialist selection
""")


# ================================================================
# Run all demos
# ================================================================

if __name__ == "__main__":
    print("\nADK AgentTool Delegation")
    print("=" * 60)
    print(f"Model: {model}\n")

    async def run_all():
        await demo_agent_delegation()
        await demo_comparison()

    asyncio.run(run_all())

    print("=" * 60)
    print("KEY TAKEAWAYS:")
    print("  1. AgentTool wraps an agent so it can be used as a tool")
    print("  2. The parent agent DECIDES which specialist to call")
    print("  3. Specialists can have their own tools")
    print("  4. Different from SequentialAgent (fixed order vs dynamic)")
    print("  5. Great for building coordinator/specialist architectures")
    print("=" * 60)
