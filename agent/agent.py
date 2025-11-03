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

weather_agent = Agent(
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

print(f"Agent '{weather_agent.name}' created with model '{weather_agent.model}'.")


# Session Management
# Concept: SessionService stores conversation history and state
# InMemorySessionService is used as a simple non-persistent storage for this project
session_service = InMemorySessionService()

# Define constants for the identifying the interaction context
APP_NAME = "weather_agent_app"
USER_ID = "user_001"
SESSION_ID = "session_001"

# Create the specific session where the convo will happen
async def create_session():
    return await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )

session = asyncio.run(create_session())

print(f"Session created: App='{APP_NAME}', User='{USER_ID}', Session='{SESSION_ID}'")

# Initiate runner
runner = Runner(
    agent=weather_agent,
    app_name=APP_NAME,
    session_service=session_service
)
print(f"Runner initialized for agent '{runner.agent.name}'.")


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
        # print all events during execution
        # print(f" [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")

        # is_final_response() marks the concluding message for the turn
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate:
                final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
            break

    print(f"<<< Agent response: {final_response_text}")

# run the initial conversation
# define async function to await interaction helper
async def run_conversation():
    await call_agent_async("What is the weather in Nairobi?", runner=runner, user_id=USER_ID, session_id=SESSION_ID)
    await call_agent_async("What is the weather in New York?", runner=runner, user_id=USER_ID, session_id=SESSION_ID)
    await call_agent_async("What is the weather in London?", runner=runner, user_id=USER_ID, session_id=SESSION_ID)


if __name__ == "__main__":
    try:
        # asyncio.run(create_session())
        asyncio.run(run_conversation())
    except Exception as e:
        print(f"Error occurred: {e}")

