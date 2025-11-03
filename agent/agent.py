import os
import asyncio
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
    print(f"--- Tool: get_weather called for {city} city ---")
    city_normalized = city.lower()

    # mock weather data
    mock_weather_data = {
        "nairobi": {"status": "success", "report": "The weather in Nairobi is sunny with a temperature of 25 degrees Celsius"},
        "new york": {"status": "success", "report": "The weather in New York is cloudy with a temperature of 15 degrees Celsius"},
        "london": {"status": "success", "report": "The weather in London is rainy with a temperature of 20 degrees Celsius"},
        "paris": {"status": "success", "report": "The weather in Paris is cloudy with a temperature of 10 degrees Celsius"},
        "cape town": {"status": "success", "report": "The weather in Cape Town is cloudy with a temperature of 15 degrees Celsius"},
    }

    if city_normalized in mock_weather_data:
        return mock_weather_data[city_normalized]
    else:
        return {"status": "error", "error_message": f"Weather information for {city} is not available."}

print(get_weather("nairobi"))
print(get_weather("new york"))

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
    tools=[get_weather], # Pass the function directly
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

