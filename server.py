import logging
import os
import random
import sys
import requests
from mcp.server.fastmcp import FastMCP
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo
from dotenv import load_dotenv
import os

load_dotenv()

name = "demo-mcp-server"
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(name)

mcp = FastMCP(name)

model_client = OpenAIChatCompletionClient(
    model="gemini-2.0-flash-lite",
    model_info=ModelInfo(
        vision=True,
        function_calling=True,
        json_output=True,
        family="unknown",
        structured_output=True
    ),
    api_key=os.getenv("API_KEY"),
)

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    logger.info(f"Tool called: add({a}, {b})")
    return a + b

@mcp.tool()
async def java_code(desc: str) -> str: 
    """Writes Java code based on given description"""
    agent = AssistantAgent(
        name="AGENT",
        model_client=model_client,
        system_message="You are a java developer who returns only java code without any explanations.",
        reflect_on_tool_use=True,
        model_client_stream=True, 
    )
    task = f"Write Java code for: {desc}. Only return Java code, with no explanations."
    result = await agent.run(task=task)
    
    messages = result.get("messages", []) if isinstance(result, dict) else getattr(result, "messages", [])
    if not messages:
        return ""
    last_content = messages[-1].content if hasattr(messages[-1], "content") else messages[-1].get("content", "")
    
    if last_content.startswith("``````"):
        last_content = last_content[len("``````")]
    
    return last_content.strip()


@mcp.tool()
async def java_to_py_code_converter(javaCode: str) -> str: 
    """converts java code to python code"""
    agent = AssistantAgent(
        name="AGENT",
        model_client=model_client,
        system_message="You are a code converter who takes java code and return python version of the given java code.",
        reflect_on_tool_use=True,
        model_client_stream=True, 
    )
    task = f"Convert the given java code: {javaCode} to Python code. Only return Python code, with no explanations."
    result = await agent.run(task=task)
    
    messages = result.get("messages", []) if isinstance(result, dict) else getattr(result, "messages", [])
    if not messages:
        return ""
    last_content = messages[-1].content if hasattr(messages[-1], "content") else messages[-1].get("content", "")
    
    if last_content.startswith("``````"):
        last_content = last_content[len("``````")]
    
    return last_content.strip()


@mcp.tool()
def get_secret_word() -> str:
    """Get a random secret word"""
    logger.info("Tool called: get_secret_word()")
    return random.choice(["Shata", "Lowda", "Thika"])

@mcp.tool()
def get_current_weather(city: str) -> str:
    """Get current weather for a city"""
    logger.info(f"Tool called: get_current_weather({city})")
    try:
        endpoint = "https://wttr.in"
        response = requests.get(f"{endpoint}/{city}", timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        logger.error(f"Error fetching weather data: {str(e)}")
        return f"Error fetching weather data: {str(e)}"

if __name__ == "__main__":
    logger.info("Starting MCP Server on stdio transport...")
    try:
        mcp.run(transport="stdio")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        sys.exit(1)
    finally:
        logger.info("Server terminated")
