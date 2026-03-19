"""
ADK Reference Agent — Clean, Commented Implementation
========================================================
A complete, production-style ADK tool-using agent that you
can use as a reference or starting point for your own agents.

Features:
  - Weather lookup (Open-Meteo API, free, no key needed)
  - Calculator (safe math evaluation)
  - Automatic tool-calling loop (ADK handles it)
  - Session persistence across conversation turns
  - Async event-driven architecture

This file exports create_agent() and create_runner() for reuse.

Requires: GOOGLE_API_KEY in your environment.

Run: python week-02-framework-basics/adk/simple_llm_agent.py
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
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


# ═══════════════════════════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════════════════════════
# ADK tools are plain Python functions — no decorators needed.
# ADK reads the function name, type hints, and docstring to
# generate the tool schema automatically.
#
# IMPORTANT: Type hints and docstrings are REQUIRED for ADK tools.

def get_weather(city: str) -> str:
    """Get the current weather for a city.
    Returns temperature and wind speed.
    Use this when the user asks about weather conditions.
    """
    # Primary: Open-Meteo (free, no API key)
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=10,
        ).json()

        if geo.get("results"):
            loc = geo["results"][0]
            weather = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": loc["latitude"],
                    "longitude": loc["longitude"],
                    "current_weather": True,
                },
                timeout=10,
            ).json()
            cw = weather["current_weather"]
            return (f"Weather in {loc['name']}: {cw['temperature']}°C, "
                    f"wind {cw['windspeed']} km/h")
    except Exception:
        pass

    # Fallback: wttr.in
    try:
        resp = requests.get(
            f"https://wttr.in/{city}?format=%t+%C",
            timeout=5,
            headers={"User-Agent": "curl/7.68.0"},
        )
        if resp.status_code == 200 and "Unknown" not in resp.text:
            return f"Weather in {city}: {resp.text.strip()}"
    except Exception:
        pass

    return f"Could not fetch weather for {city} (APIs unavailable)"


def calculate(expression: str) -> str:
    """Evaluate a math expression. Supports +, -, *, /, ** (power), and parentheses.
    Examples: '15 * 7', '2 ** 10', '(100 + 50) / 3'
    """
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: Only numbers and operators (+-*/.) are allowed"
        result = eval(expression)
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return f"Error: Division by zero in '{expression}'"
    except Exception as e:
        return f"Error: {e}"


# ═══════════════════════════════════════════════════════════════
# AGENT & RUNNER CREATION
# ═══════════════════════════════════════════════════════════════

def create_agent() -> LlmAgent:
    """Create and return an ADK agent with weather and calculator tools.

    Returns:
        An LlmAgent instance ready to be used with a Runner.
    """
    return LlmAgent(
        name="reference_agent",
        model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
        instruction="""You are a helpful assistant with access to tools.

        Available tools:
        - get_weather: Look up current weather for any city
        - calculate: Evaluate math expressions (use Python syntax)

        Guidelines:
        - Use tools when the question requires real-time data or calculations
        - If a tool returns an error, explain the issue to the user
        - For general knowledge questions, answer directly without tools
        - Be concise and helpful""",
        tools=[get_weather, calculate],
        description="A helpful assistant that can check weather and do math.",
    )


def create_runner(agent: LlmAgent = None) -> tuple:
    """Create a Runner and SessionService for the given agent.

    Args:
        agent: An LlmAgent instance. If None, creates one via create_agent().

    Returns:
        Tuple of (runner, session_service)
    """
    if agent is None:
        agent = create_agent()

    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name="reference_app",
        session_service=session_service,
    )
    return runner, session_service


# ═══════════════════════════════════════════════════════════════
# HELPER: Send a query and get a response
# ═══════════════════════════════════════════════════════════════

async def ask(runner: Runner, session_service: InMemorySessionService,
              session_id: str, query: str) -> str:
    """Send a query to the agent and return the response text.

    Args:
        runner: The ADK Runner instance
        session_service: The session service (for session management)
        session_id: The session ID to use (for multi-turn conversations)
        query: The user's question

    Returns:
        The agent's response text
    """
    response_text = ""
    async for event in runner.run_async(
        user_id="user",
        session_id=session_id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=query)],
        ),
    ):
        if event.is_final_response():
            response_text = event.content.parts[0].text

    return response_text


# ═══════════════════════════════════════════════════════════════
# INTERACTIVE MODE
# ═══════════════════════════════════════════════════════════════

async def main():
    """Run the agent in interactive mode."""
    model = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
    print(f"ADK Reference Agent (model: {model})")
    print("=" * 60)
    print("Tools: get_weather, calculate")
    print("Type 'quit' to exit\n")

    runner, session_service = create_runner()

    # Create a persistent session for the conversation
    session = await session_service.create_session(
        app_name="reference_app", user_id="user"
    )

    # Demo queries
    demo_queries = [
        "What's the weather in Tokyo?",
        "Calculate (15 + 27) * 3",
    ]

    for query in demo_queries:
        print(f"Query: {query}")
        try:
            response = await ask(runner, session_service, session.id, query)
            print(f"Agent: {response}\n")
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}\n")

    # Interactive loop
    print("-" * 60)
    print("Your turn! Ask anything (weather, math, or general questions):\n")
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if not user_input:
                continue
            response = await ask(runner, session_service, session.id, user_input)
            print(f"Agent: {response}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
