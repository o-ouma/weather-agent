import os
import asyncio
import urllib.request
import urllib.parse
import json
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from dotenv import load_dotenv, find_dotenv
import logging
logging.basicConfig(level=logging.ERROR)


# Helper to detect rate-limit-style errors across different LLM wrappers
def _is_rate_limit_error(exc: Exception) -> bool:
    """Heuristically determine if an exception represents a rate limit error.

    We try to detect common exception classes (e.g., litellm.RateLimitError) if
    available, otherwise fall back to searching for 'RateLimit' in the exception
    type name or message.
    """
    if exc is None:
        return False
    # Check type name first
    tname = type(exc).__name__
    if "RateLimit" in tname or "RateLimit" in str(exc):
        return True
    # Try to import litellm if present and check explicitly
    try:
        import litellm
        RateLimitErr = getattr(litellm, "RateLimitError", None)
        if RateLimitErr and isinstance(exc, RateLimitErr):
            return True
    except Exception:
        # litellm not available or import failed — ignore
        pass
    return False


# Load variables from .env file if present
load_dotenv(find_dotenv())

def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a given city.

    Args:
        city (str): Name of city

    Returns:
        dict: A dictionary containing the weather information
              Includes status key
    """
    if not city or not city.strip():
        return {"status": "error", "error_message": "City must be non-empty string."}

    # wttr.in provides a simple json api at /{city}?format=j1
    encoded_city = urllib.parse.quote(city)
    url = f"https://wttr.in/{encoded_city}?format=j1"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            status = getattr(response, "status", None)
            if status is not None and status != 200:
                return {"status": "error", "error_message": f"Error fetching weather data: {status}."}
            body = response.read().decode("utf-8")
            data = json.loads(body)
    except Exception as e:
        return {"status": "error", "error_message": f"Error fetching weather data: {e}."}

    # Parse the common fields from wttr.in response
    try:
        current = data.get("current_condition", [])[0]
        temp_c = current.get("temp_C", None)
        desc = current.get("weatherDesc", [{}])[0].get("value")
        humidity = current.get("humidity", None)
        feels_like_c = current.get("feelslike_C", None)

        report = (
            f"Weather in {city} is {desc}, temperature {temp_c}\u00b0c (feels like {feels_like_c}\u00b0c), humidity {humidity}%."
        )

        return {"status": "success", "report": report, "data": data}
    except Exception as e:
        return {"status": "error", "error_message": f"Error parsing weather data: {e}."}

# Model specification
GEMINI_AGENT_MODEL = os.environ.get("AGENT_MODEL")
MODEL_GPT_40 = os.environ.get("MODEL_GPT")
MODEL_CLAUDE_SONNET = os.environ.get("CLAUDE_SONNET")


# session helper
async def create_session_for(session_service, app_name, user_id, session_id):
    return await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id
    )


# define agent interaction function
async def call_agent_async(query: str, runner, user_id, session_id):
    """Sends a query to the agent and retrieves the response."""
    print(f"--- Agent interaction started for query: '{query}' ---")

    # prepare user message in adk format
    content = types.Content(role='user', parts=[types.Part(text=query)])

    final_response_text = "Agent did not produce a final response."

    # run_async executes the agent logic and yields events
    # we iterate through events to find the final answer
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        # is_final_response() marks the concluding message for the turn
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate:
                final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
            break

    print(f"<<< Agent response: {final_response_text}")

root_agent = Agent(
    name="weather_agent_v1",
    model=GEMINI_AGENT_MODEL,
    description="Provides weather information for specific cities.",
    instruction="You are a helpful weather assistant. "
                "When the user asks for the weather in a specific city, "
                "use the 'get_weather' tool to find the information. "
                "If the tool returns an error, inform the user politely. "
                "If the tool is successful, present the weather report clearly.",
    tools=[get_weather],
)

print(f"Agent '{root_agent.name}' created with model '{root_agent.model}'.")


# Define constants for the identifying the interaction context
APP_NAME = "weather_agent_app"
USER_ID = "user_001"
SESSION_ID = "session_001"

# Session / runner placeholders (defer actual async initialization)
# We avoid calling asyncio.run() at module import time so this module can be
# imported inside an existing event loop (e.g., when run by `adk web`).
session_service = None
session = None
runner = None


async def init_default_runner():
    """Create the default in-memory session and Runner for the non-GPT agent.

    This is async so callers running inside an event loop (like adk web)
    can await it instead of forcing a new loop with asyncio.run().
    """
    global session_service, session, runner
    if session_service is None:
        session_service = InMemorySessionService()

    # create session asynchronously
    session = await create_session_for(session_service, APP_NAME, USER_ID, SESSION_ID)
    print(f"Session created: App='{APP_NAME}', User='{USER_ID}', Session='{SESSION_ID}'")

    # create runner
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)
    print(f"Runner initialized for agent '{runner.agent.name}'.")

#
# # Add gpt agent (defer session/runner initialization)
# weather_agent_gpt = None
# runner_gpt = None
#
# # default placeholders for GPT identifiers (set if GPT agent creation succeeds)
# APP_NAME_GPT = USER_ID_GPT = SESSION_ID_GPT = None
#
# if MODEL_GPT_40:
#     try:
#         # Create GPT-based agent only if the MODEL_GPT environment variable is set.
#         # Pass the variable (not the literal string) into LiteLlm.
#         weather_agent_gpt = Agent(
#             name="weather_agent_gpt",
#             model=LiteLlm(model=MODEL_GPT_40),
#             description="Provides weather information for specific cities (using GPT-like model).",
#             instruction="You are a helpful weather assistant. "
#                         "When the user asks for the weather in a specific city, "
#                         "use the 'get_weather' tool to find the information. "
#                         "If the tool returns an error, inform the user politely. "
#                         "If the tool is successful, present the weather report clearly.",
#             tools=[get_weather],
#         )
#         print(f"Agent '{weather_agent_gpt.name}' created with model '{MODEL_GPT_40}'.")
#
#
#         APP_NAME_GPT = "weather_agent_app_gpt"
#         USER_ID_GPT = "user_002_gpt"
#         SESSION_ID_GPT = "session_002_gpt"
#
#         # session_service_gpt, session_gpt and runner_gpt will be created by init_gpt_runner()
#
#     except Exception as e:
#         print(
#             f"Could not create or initialize GPT agent with model '{MODEL_GPT_40}'.\n"
#             f"Error: {e}\n"
#             "Action: Ensure the environment variable MODEL_GPT is set to a supported model/provider string\n"
#             "(examples: 'gpt-4o' or 'huggingface/starcoder' for Hugging Face inference).\n"
#             "If your LLM provider requires extra credentials or a provider field, set those accordingly."
#         )
# else:
#     print(
#         "Skipping GPT agent creation: environment variable MODEL_GPT is not set.\n"
#         "Set MODEL_GPT to the model/provider you want to use (for example: MODEL_GPT=gpt-4o) "
#         "or use a Hugging Face model string like 'huggingface/starcoder'."
#     )
#
#
# async def init_gpt_runner():
#     """Async initializer for the GPT runner/session. Call only if MODEL_GPT_40 is set."""
#     global runner_gpt, session_gpt, session_service_gpt
#     if not MODEL_GPT_40 or weather_agent_gpt is None:
#         return
#
#     # create session service and session
#     session_service_gpt = InMemorySessionService()
#     session_gpt = await create_session_for(session_service_gpt, APP_NAME_GPT, USER_ID_GPT, SESSION_ID_GPT)
#     print(f"Session created: App='{APP_NAME_GPT}', User='{USER_ID_GPT}', Session='{SESSION_ID_GPT}'")
#
#     runner_gpt = Runner(agent=weather_agent_gpt, app_name=APP_NAME_GPT, session_service=session_service_gpt)
#     print(f"Runner initialized for agent '{runner_gpt.agent.name}'.")
#
#     return


# If the module is imported into a running event loop (e.g., by `adk web`),
# schedule initialization tasks instead of calling asyncio.run(). This avoids
# the "asyncio.run() cannot be called from a running event loop" error.
try:
    _loop = asyncio.get_running_loop()
except RuntimeError:
    _loop = None

if _loop and not _loop.is_closed():
    # schedule initialization in the background when ADK imports the module
    try:
        _loop.create_task(init_default_runner())
        if MODEL_GPT_40:
            _loop.create_task(init_gpt_runner())
    except Exception:
        # If scheduling fails, silently continue; main() will initialize when run.
        pass


# run the initial conversation
# define async function to await interaction helper
async def run_conversation():
    # Ensure default runner/session are initialized before invoking the agent
    if runner is None:
        await init_default_runner()

    if runner is None:
        raise RuntimeError("Failed to initialize default runner; cannot run conversation.")

    await call_agent_async("What is the weather in Nairobi?", runner=runner, user_id=USER_ID, session_id=SESSION_ID)
    await call_agent_async("What is the weather in New York?", runner=runner, user_id=USER_ID, session_id=SESSION_ID)
    await call_agent_async("What is the weather in London?", runner=runner, user_id=USER_ID, session_id=SESSION_ID)


# Consolidate the two entry points under a single main() to avoid duplication.
async def main():
    try:
        # Initialize default runner (async-safe)
        await init_default_runner()

        # Run the default agent conversation
        await run_conversation()

        # If a GPT model is configured, try to initialize and test it
        # if MODEL_GPT_40:
        #     await init_gpt_runner()
        #     if runner_gpt is not None and USER_ID_GPT and SESSION_ID_GPT:
        #         print("\n---Testing GPT Agent (from main)---")
        #         await call_agent_async("What is the weather in Nairobi?", runner=runner_gpt, user_id=USER_ID_GPT, session_id=SESSION_ID_GPT)

    except Exception as e:
        # Provide a clear message and re-raise for visibility when running interactively
        print(f"Error occurred in main(): {e}")
        raise


if __name__ == "__main__":
    # Single entry point: run the consolidated main coroutine
    asyncio.run(main())
