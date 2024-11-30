import os
import sys
from typing import Optional
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import asyncio

from browser_use import Agent, BrowserConfig
from browser_use.controller.service import Controller

def get_llm(provider: str):
    """
    Initialize and return the specified language model.
    
    Args:
        provider (str): The provider name ('anthropic', 'openai', or 'gemini')
        
    Returns:
        The initialized language model
    
    Raises:
        ValueError: If an unsupported provider is specified
    """
    if provider == 'anthropic':
        if 'ANTHROPIC_API_KEY' not in os.environ:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required for Anthropic")
        return ChatAnthropic(
            model_name='claude-3-5-sonnet-20240620',
            timeout=25,
            stop=None,
            temperature=0.0
        )
    elif provider == 'openai':
        if 'OPENAI_API_KEY' not in os.environ:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI")
        return ChatOpenAI(
            model='gpt-4',
            temperature=0.0
        )
    elif provider == 'gemini':
        if 'GOOGLE_API_KEY' not in os.environ:
            raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini")
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.0,
            
        )
    else:
        raise ValueError(f'Unsupported provider: {provider}')

def setup_argument_parser():
    """Set up and return the argument parser"""
    parser = argparse.ArgumentParser(description='Run browser agent with specified LLM provider')
    parser.add_argument(
        'query',
        type=str,
        help='The query to process'
    )
    parser.add_argument(
        '--provider',
        type=str,
        choices=['openai', 'anthropic', 'gemini'],
        default='openai',
        help='The model provider to use (default: openai)'
    )
    parser.add_argument(
        '--save-path',
        type=str,
        help='Path to save the conversation (optional)',
        default=None
    )
    parser.add_argument(
        '--keep-browser',
        action='store_true',
        help='Keep the browser open after completion'
    )
    return parser

async def main():
    """Main function to run the browser agent"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    try:
        llm = get_llm(args.provider)
        
        # Initialize the browser agent
        agent = Agent(
            task=args.query,
            llm=llm,
            controller=Controller(
                browser_config=BrowserConfig(keep_open=args.keep_browser)
            ),
            save_conversation_path=args.save_path
        )
        
        # Run the agent
        await agent.run()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())