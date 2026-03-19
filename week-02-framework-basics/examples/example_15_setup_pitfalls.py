"""
Example 15: Common Setup Pitfalls
====================================
When setting up LangGraph and ADK, there are several common mistakes
that waste hours of debugging. This example documents each pitfall,
shows what goes wrong, and provides the fix.

No API keys needed for the demonstrations (uses mock examples).

Run: python week-02-framework-basics/examples/example_15_setup_pitfalls.py
"""

import os
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()


def pitfall_1_missing_env():
    """Pitfall 1: Missing or wrong .env configuration."""
    print("=" * 60)
    print("PITFALL 1: Missing or Wrong .env Configuration")
    print("=" * 60)
    print("""
  SYMPTOM:
    AuthenticationError: Invalid API key
    ValueError: GOOGLE_API_KEY not set

  CAUSE:
    - .env file not created (only .env.example exists)
    - .env file in wrong location
    - API key has extra spaces or quotes
    - Wrong variable name

  FIX:
    1. Copy the template:
       cp config/.env.example config/.env

    2. Edit config/.env with your ACTUAL keys:
       GROQ_API_KEY=gsk_abc123...     (no quotes!)
       GOOGLE_API_KEY=AIza...          (no quotes!)
       OPENAI_API_KEY=sk-...           (no quotes!)

    3. Make sure load_dotenv uses the RIGHT path:
       load_dotenv("config/.env")      # Correct
       load_dotenv(".env")             # WRONG -- different file!

  QUICK CHECK:""")

    # Show how to verify your keys are loaded
    groq_key = os.getenv("GROQ_API_KEY", "")
    google_key = os.getenv("GOOGLE_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    provider = os.getenv("LLM_PROVIDER", "groq")

    print(f"    LLM_PROVIDER:   {provider}")
    print(f"    GROQ_API_KEY:   {'set (' + groq_key[:8] + '...)' if groq_key else 'NOT SET'}")
    print(f"    GOOGLE_API_KEY: {'set (' + google_key[:8] + '...)' if google_key else 'NOT SET'}")
    print(f"    OPENAI_API_KEY: {'set (' + openai_key[:8] + '...)' if openai_key else 'NOT SET'}")
    print()


def pitfall_2_import_errors():
    """Pitfall 2: Import errors from missing packages."""
    print("=" * 60)
    print("PITFALL 2: Import Errors (Missing Packages)")
    print("=" * 60)
    print("""
  SYMPTOM:
    ModuleNotFoundError: No module named 'langchain_groq'
    ModuleNotFoundError: No module named 'google.adk'

  CAUSE:
    - Forgot to install requirements
    - Wrong virtual environment activated
    - Package name different from import name

  FIX:
    pip install -r requirements.txt

  COMMON PACKAGE MAPPING:
    Import name              Package name
    ---------                ------------
    langchain_groq           langchain-groq
    langchain_openai         langchain-openai
    langchain_google_genai   langchain-google-genai
    langgraph                langgraph
    google.adk               google-adk
    google.genai             google-genai
    phoenix                  arize-phoenix

  QUICK CHECK:""")

    packages = [
        ("langgraph", "langgraph"),
        ("langchain_groq", "langchain-groq"),
        ("langchain_openai", "langchain-openai"),
    ]

    for import_name, pip_name in packages:
        try:
            __import__(import_name)
            print(f"    {import_name:<25} installed")
        except ImportError:
            print(f"    {import_name:<25} NOT INSTALLED (pip install {pip_name})")

    # Check ADK separately (nested import)
    try:
        from google.adk.agents import LlmAgent
        print(f"    {'google.adk':<25} installed")
    except ImportError:
        print(f"    {'google.adk':<25} NOT INSTALLED (pip install google-adk)")

    print()


def pitfall_3_groq_tool_errors():
    """Pitfall 3: Groq/Llama malformed tool calls."""
    print("=" * 60)
    print("PITFALL 3: Groq/Llama Malformed Tool Calls")
    print("=" * 60)
    print("""
  SYMPTOM:
    BadRequestError: 400 tool_use_failed
    "Expected a tool call but got malformed JSON"

  CAUSE:
    Groq's Llama models sometimes generate invalid tool call JSON.
    This is a known issue with open-source models -- they're not
    as reliable at structured output as GPT-4 or Gemini.

  FIX:
    Always wrap LLM calls in retry logic:

    def agent_node(state):
        for attempt in range(3):          # Retry up to 3 times
            try:
                response = llm_with_tools.invoke(state["messages"])
                return {"messages": [response]}
            except Exception as e:
                if attempt < 2:           # Don't raise on first failures
                    continue
                raise                     # Give up after 3 tries

  WHY 3 RETRIES?
    - 1st attempt fails ~10-20% of the time
    - 2nd attempt usually succeeds
    - 3rd attempt is a safety net
    - If all 3 fail, the model is likely confused about the tool schema
""")


def pitfall_4_adk_async():
    """Pitfall 4: ADK requires async/await."""
    print("=" * 60)
    print("PITFALL 4: ADK Requires async/await")
    print("=" * 60)
    print("""
  SYMPTOM:
    TypeError: object async_generator can't be used in 'await' expression
    RuntimeError: This event loop is already running

  CAUSE:
    ADK's runner.run_async() is an async generator.
    You MUST use async for, not a regular for loop.

  WRONG:
    # This crashes!
    for event in runner.run_async(...):
        print(event)

  CORRECT:
    # Use async for
    async for event in runner.run_async(...):
        if event.is_final_response():
            print(event.content.parts[0].text)

  ALSO CORRECT (running from sync code):
    import asyncio

    async def ask(query):
        async for event in runner.run_async(...):
            if event.is_final_response():
                return event.content.parts[0].text

    # Call from sync context
    result = asyncio.run(ask("What is AI?"))

  JUPYTER NOTEBOOK NOTE:
    Jupyter already runs an event loop, so use:
      await ask("What is AI?")
    instead of:
      asyncio.run(ask("What is AI?"))
""")


def pitfall_5_state_mistakes():
    """Pitfall 5: LangGraph state management mistakes."""
    print("=" * 60)
    print("PITFALL 5: LangGraph State Mistakes")
    print("=" * 60)
    print("""
  MISTAKE 1: Forgetting add_messages reducer
  -------------------------------------------
  WRONG:
    class State(TypedDict):
        messages: list           # Plain list -- gets REPLACED each node!

  CORRECT:
    class State(TypedDict):
        messages: Annotated[list, add_messages]  # APPENDS messages

  Without add_messages, each node REPLACES the message history,
  and the agent loses all context from previous turns.

  MISTAKE 2: Not providing all initial state fields
  ---------------------------------------------------
  WRONG:
    app.invoke({"messages": [HumanMessage(content="hi")]})
    # KeyError: 'topic' -- if your state has a 'topic' field

  CORRECT:
    app.invoke({
        "messages": [HumanMessage(content="hi")],
        "topic": "",           # Provide defaults for ALL fields
        "iteration": 0,
    })

  MISTAKE 3: Returning full state from nodes
  -------------------------------------------
  WRONG:
    def my_node(state):
        return state           # Returns EVERYTHING -- may cause issues

  CORRECT:
    def my_node(state):
        return {"summary": "..."}  # Return ONLY changed fields
""")


def pitfall_6_provider_mismatch():
    """Pitfall 6: Using wrong model with wrong framework."""
    print("=" * 60)
    print("PITFALL 6: Framework-Provider Mismatch")
    print("=" * 60)
    print("""
  ADK ONLY works with Google Gemini models:
    LlmAgent(model="gemini-2.0-flash")     # Works
    LlmAgent(model="gpt-4o-mini")          # FAILS
    LlmAgent(model="llama-3.3-70b")        # FAILS

  LangGraph works with ANY LangChain-supported LLM:
    ChatGroq(model="llama-3.3-70b-versatile")   # Works
    ChatOpenAI(model="gpt-4o-mini")              # Works
    ChatGoogleGenerativeAI(model="gemini-...")    # Works

  BEST PRACTICE:
    - Use LangGraph when you need provider flexibility
    - Use ADK when you're committed to the Google ecosystem
    - Always check LLM_PROVIDER env var for portability:

      provider = os.getenv("LLM_PROVIDER", "groq").lower()
      if provider == "groq":
          llm = ChatGroq(model=...)
      else:
          llm = ChatOpenAI(model=...)
""")


def pitfall_7_windows_issues():
    """Pitfall 7: Windows-specific issues."""
    print("=" * 60)
    print("PITFALL 7: Windows-Specific Issues")
    print("=" * 60)
    print("""
  ISSUE 1: Unicode encoding errors
    UnicodeEncodeError: 'charmap' codec can't encode character

    FIX: Add at the top of your script:
      import sys
      if sys.platform == "win32":
          sys.stdout.reconfigure(encoding="utf-8", errors="replace")

  ISSUE 2: Phoenix temp file cleanup errors on exit
    PermissionError: [WinError 32] The process cannot access the file

    FIX: Use os._exit(0) instead of sys.exit(0) at the end:
      import os
      os._exit(0)

  ISSUE 3: asyncio event loop issues
    RuntimeError: Event loop is closed

    FIX: Use asyncio.run() from the main thread only:
      if __name__ == "__main__":
          asyncio.run(main())

  ISSUE 4: Path separators
    Use forward slashes or os.path.join:
      WRONG:  "config\\.env"
      RIGHT:  "config/.env"   (works on all platforms)
""")


# ================================================================
# Run all pitfalls
# ================================================================

if __name__ == "__main__":
    print("\nCommon Setup Pitfalls and How to Fix Them")
    print("=" * 60)
    print("These are the most common issues students hit in Week 2.\n")

    pitfall_1_missing_env()
    pitfall_2_import_errors()
    pitfall_3_groq_tool_errors()
    pitfall_4_adk_async()
    pitfall_5_state_mistakes()
    pitfall_6_provider_mismatch()
    pitfall_7_windows_issues()

    print("=" * 60)
    print("SUMMARY OF FIXES:")
    print("  1. .env: Copy template, no quotes around keys")
    print("  2. Imports: pip install -r requirements.txt")
    print("  3. Groq tools: Always retry up to 3 times")
    print("  4. ADK: Use async for, not regular for")
    print("  5. State: Use add_messages, provide all defaults")
    print("  6. Providers: ADK = Gemini only, LangGraph = any LLM")
    print("  7. Windows: UTF-8 encoding, os._exit for Phoenix")
    print("=" * 60)
