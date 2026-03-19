"""
Example 11: ADK YAML Configuration
=====================================
ADK agents can be configured using YAML files instead of Python code.
This is useful for:
  - Separating agent behavior from code
  - Quick iteration on prompts without code changes
  - Non-developers editing agent instructions
  - Version-controlling agent configurations

This example shows:
  1. How agent.yaml files are structured
  2. Loading agents from YAML configuration
  3. Comparing YAML config vs Python code

Note: This example demonstrates the YAML format and loads configs
manually, since the full ADK YAML loading requires specific project
structure. The concepts apply to the standard ADK project layout.

Requires: GOOGLE_API_KEY in your environment.

Run: python week-02-framework-basics/examples/example_11_adk_yaml_config.py
"""

import asyncio
import logging
import os
import yaml
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

logging.getLogger("google_genai.types").setLevel(logging.ERROR)

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


# ================================================================
# PART 1: Understanding the YAML Format
# ================================================================

def demo_yaml_format():
    """Show and explain the ADK agent YAML format."""
    print("=" * 60)
    print("PART 1: ADK Agent YAML Format")
    print("=" * 60)

    # This is what an agent.yaml file looks like in an ADK project
    yaml_config = """
# agent.yaml -- ADK agent configuration
# Place this in your agent's directory alongside __init__.py

name: research_assistant
model: gemini-2.0-flash
description: A helpful research assistant that answers questions.

instruction: |
  You are a helpful research assistant.

  Guidelines:
  - Answer questions clearly and concisely
  - If you don't know something, say so
  - Provide sources when possible
  - Keep responses under 200 words

# Tools are referenced by their Python function names
# The functions must be importable from the agent's module
tools:
  - search_web
  - calculate

# Generation parameters
generate_content_config:
  temperature: 0.7
  max_output_tokens: 500
"""

    print(f"\nExample agent.yaml:\n")
    print(yaml_config)

    # Parse the YAML to show it's just a dictionary
    config = yaml.safe_load(yaml_config)
    print("Parsed as Python dict:")
    for key, value in config.items():
        if key == "instruction":
            print(f"  {key}: '{value[:50]}...'")
        else:
            print(f"  {key}: {value}")
    print()


# ================================================================
# PART 2: Creating an Agent from YAML Config
# ================================================================

async def demo_yaml_to_agent():
    """Load agent configuration from YAML and create an LlmAgent."""
    print("=" * 60)
    print("PART 2: Creating an Agent from YAML Config")
    print("=" * 60)

    # Define config as YAML string (in practice, load from a file)
    yaml_config = """
name: tutor_agent
model: gemini-2.0-flash
description: An AI tutor that explains concepts clearly.
instruction: |
  You are a patient AI tutor.
  - Explain concepts simply, as if to a beginner
  - Use analogies when helpful
  - Keep answers to 2-3 sentences
  - End with a question to check understanding
generate_content_config:
  temperature: 0.8
  max_output_tokens: 300
"""

    # Parse YAML config
    config = yaml.safe_load(yaml_config)

    # Build generation config if provided
    gen_config = None
    if "generate_content_config" in config:
        gen_config = types.GenerateContentConfig(
            temperature=config["generate_content_config"].get("temperature", 0.7),
            max_output_tokens=config["generate_content_config"].get("max_output_tokens", 500),
        )

    # Create agent from config -- this is what ADK does internally
    agent = LlmAgent(
        name=config["name"],
        model=os.getenv("GOOGLE_MODEL", config.get("model", "gemini-2.0-flash")),
        instruction=config.get("instruction", ""),
        description=config.get("description", ""),
        generate_content_config=gen_config,
    )

    print(f"\nCreated agent from YAML:")
    print(f"  name: {config['name']}")
    print(f"  model: {config.get('model')}")
    print(f"  temperature: {config.get('generate_content_config', {}).get('temperature')}")
    print(f"  instruction: '{config['instruction'][:60]}...'\n")

    # Run a query
    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name="yaml_demo",
        session_service=session_service,
    )

    session = await session_service.create_session(
        app_name="yaml_demo", user_id="user1"
    )

    query = "What is a state machine?"
    print(f"Query: {query}")

    async for event in runner.run_async(
        user_id="user1",
        session_id=session.id,
        new_message=types.Content(
            role="user", parts=[types.Part(text=query)]
        ),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            print(f"Agent: {event.content.parts[0].text}")
    print()


# ================================================================
# PART 3: YAML vs Python -- When to Use Each
# ================================================================

def demo_yaml_vs_python():
    """Compare YAML config vs Python code approaches."""
    print("=" * 60)
    print("PART 3: YAML vs Python Config")
    print("=" * 60)

    print("""
  YAML Configuration                  Python Code
  --------------------------------    --------------------------------
  agent.yaml:                         agent.py:
    name: my_agent                      agent = LlmAgent(
    model: gemini-2.0-flash                 name="my_agent",
    instruction: |                          model="gemini-2.0-flash",
      You are helpful.                      instruction="You are helpful.",
    tools:                                  tools=[search, calc],
      - search                          )
      - calc

  WHEN TO USE YAML:                   WHEN TO USE PYTHON:
  + Non-developers edit prompts       + Dynamic configuration
  + Quick prompt iteration            + Complex tool setup
  + Config version control            + Conditional logic
  + Standard ADK project layout       + Custom agent classes
  + A/B testing different prompts     + Integration with other systems

  ADK PROJECT STRUCTURE (standard layout):
    my_agent/
      __init__.py          # Exports the agent
      agent.yaml           # Agent configuration
      tools.py             # Tool function definitions
      prompts/
        system.txt         # Long system prompts

  The standard ADK CLI (adk run) automatically loads agent.yaml
  when you run your agent project.
""")


# ================================================================
# Run all demos
# ================================================================

if __name__ == "__main__":
    print("\nADK YAML Configuration")
    print("=" * 60)

    demo_yaml_format()

    asyncio.run(demo_yaml_to_agent())

    demo_yaml_vs_python()

    print("=" * 60)
    print("KEY TAKEAWAYS:")
    print("  1. agent.yaml defines name, model, instruction, tools")
    print("  2. YAML separates config from code -- easier to iterate")
    print("  3. ADK CLI (adk run) auto-loads agent.yaml files")
    print("  4. Use YAML for prompts, Python for complex logic")
    print("  5. Both approaches create the same LlmAgent internally")
    print("=" * 60)
