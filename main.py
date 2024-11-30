from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent
import asyncio
import json

async def main():
    # Load actions from actions.json
    try:
        with open('actions.json', 'r') as f:
            actions = json.load(f)
    except FileNotFoundError:
        print("Actions file not found. Using default or empty actions.")
        actions = []

    # Initialize the agent with more explicit parameters
    agent = Agent(
        task="Open YouTube on Safari",  # Clarified task description
        llm=ChatGoogleGenerativeAI(
            model="gemini-pro",
            # Add additional configuration if needed
            temperature=0.7,
            top_p=0.8
        ),
        
    )

    try:
        result = await agent.run()
        print("Agent run result:", result)
    except Exception as e:
        print(f"Error running agent: {e}")

if __name__ == "__main__":
    asyncio.run(main())